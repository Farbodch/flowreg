"""FIX"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from diffrax import ODETerm, diffeqsolve, solver
from jax import value_and_grad
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn_lr

# def sink_div(geom):
#     """Return the Sinkhorn divergence cost and OT output given a geometry.
#     Since y is fixed, we can use static_b=True to avoid computing
#     the OT(b, b) term."""

#     ot = sinkhorn_divergence.sinkhorn_divergence(
#         geom,
#         x=geom.x,
#         y=geom.y,
#         static_b=True,
#         rank=20)
#     return ot.divergence, ot


class dataTracker:
    """FIX"""

    results = {}

    def __init__(
        self,
        source=np.array([]),
        target=np.array([]),
        sourceLabel="",
        targetLabel="",
        alphaList=None,
        iterationsList=None,
        rankList=None,
    ):
        """FIX"""

        if alphaList is None:
            alphaList = []
        if iterationsList is None:
            iterationsList = []
        if rankList is None:
            rankList = []

        self.source = source
        self.target = target
        self.sourceLabel = sourceLabel
        self.targetLabel = targetLabel
        self.alphaList = alphaList
        self.iterationsList = iterationsList
        self.rankList = rankList
        self.reset_instanceData()
        self.testNum = 0

    def reset_instanceData(self):
        """FIX"""
        self.MSE_defSource_target_ = None
        self.regLoss_ = None
        self.startTime_ = None
        self.deformed_source_ = None

        self.alpha_ = None
        self.iterations_ = None
        self.rank_ = None
        self.num_of_steps_ = None
        self.step_size_ = None

    # check to ensure not overriding an existing self._testNum!
    def set_instanceData_toDataFile(self):
        """FIX"""
        self.results[f"{self.testNum}"]["instanceMetaData"] = {
            "alpha": self.alpha_,
            "numOfOptimItrs": self.iterations_,
            "lr_sink_rank": self.rank_,
            "integratorNumSteps": self.num_of_steps_,
            "integratorStepSize": self.step_size_,
        }

    # toDataFile only called at the end or if need extraction of results into external file
    # def set_globalData_toDataFile(self):
    #     self.results['subsampledSource'] = self.source
    #     self.results['target'] = self.target
    #     self.results['sourceLabel'] = self.sourceLabel
    #     self.results['targetLabel'] = self.targetLabel


def vector_field(t, state, params):
    """Compute the time derivative of the state."""
    # V=dX/dT, dP/dT=m.dV/dT=m.V2-V1, let m=1, dT=1
    X, P = jnp.split(state, 2, axis=-1)
    V = P + params["geomGrad"]
    return jnp.concatenate([V, -V], axis=-1)


def lddmm_hamiltonian(source, target, alpha, initial_momentum, geom_grad, num_steps=10, step_size=0.1):
    """FIX"""
    # The state consists of [X, P] where X is the point cloud and P is its momentum
    initial_conditions = jnp.concatenate([source, initial_momentum], axis=-1)

    # vector_field = lambda t, y, params: vector_field(t, y, params)
    term = ODETerm(vector_field)

    # Integrate the vector_field using diffeqsolve
    states = diffeqsolve(
        terms=term,
        solver=solver.Dopri5(),
        t0=0,
        t1=num_steps * step_size,
        dt0=step_size,
        y0=initial_conditions,
        args={"geomGrad": geom_grad / 10},
    )

    # Extract the deformed source from the integrated states
    # states.ys has shape 1*numOfPts*4 where 0*x*0:2 is the position of each point & 0*x*2:4 is the momentum of each point
    deformed_source = states.ys[0, :, :2]

    # Compute the L2 loss between the deformed source and target
    loss = (jnp.mean(jnp.square(deformed_source - target)) + alpha * jnp.mean(jnp.square(initial_momentum))) ** (1 / 1)

    return loss


def sink_lr_setRank(myRank):
    """FIX"""

    def sink_lr_cost(geom):
        """Return the OT cost and OT output given a geometry"""
        ot = sinkhorn_lr.LRSinkhorn(rank=myRank, initializer="random")(linear_problem.LinearProblem(geom))
        return ot.reg_ot_cost, ot

    return sink_lr_cost


def getWeightsSetTarget(target, cost_fen, eps=0.001):
    """FIX"""
    cost_fn_vg = jax.jit(jax.value_and_grad(cost_fen, has_aux=True))
    epsilon: float = eps

    def getWeightsInner(current_source):
        """FIX"""
        geom = pointcloud.PointCloud(current_source, target, epsilon=epsilon)
        (cost, ot), geom_g = cost_fn_vg(geom)
        assert ot.converged
        # weights = ot.geoms[0].transport_from_potentials(ot.potentials[0][0],ot.potentials[0][1])
        # ots[10].geoms[0].transport_from_potentials(ots[10].potentials[0][0],ots[10].potentials[0][1])
        # transportMatrix = ot.geoms[0].transport_from_potentials(ot.potentials[0][0],ot.potentials[0][1])
        return geom_g.x

    return getWeightsInner


def runMatching(dataObj, num_steps=10, step_size=0.1):
    """FIX"""
    dataObj.reset_instanceData()

    alphaList = dataObj.alphaList
    rankList = dataObj.rankList
    iterationsList = dataObj.iterationsList

    source = dataObj.source
    target = dataObj.target

    dataObj.num_of_steps_ = num_steps
    dataObj.step_size_ = step_size

    testNum = dataObj.testNum

    # OuterForLoop Iterating over params
    for myAlpha in alphaList:
        for myRank in rankList:
            sink_lr_getCost = sink_lr_setRank(myRank)
            getSinkData = getWeightsSetTarget(target, cost_fen=sink_lr_getCost)
            for myItrNum in iterationsList:
                dataObj.testNum = testNum

                dataObj.alpha_ = myAlpha
                dataObj.rank_ = myRank
                dataObj.iterations_ = myItrNum

                optimizer = optax.adam(1e-1)

                initial_momentum = jnp.zeros_like(source)
                state = optimizer.init(initial_momentum)

                geom_grad = getSinkData(source)

                MSE_defSource_target = []
                regLoss = []

                startTime = time.strftime("%H:%M:%S", time.localtime())
                for _i in range(myItrNum):
                    initial_conditions = jnp.concatenate([source, initial_momentum], axis=-1)
                    # vector_field = lambda t, y, params: vector_field(t, y, params)
                    term = ODETerm(vector_field)
                    # Integrate the vector_field using diffeqsolve
                    states = diffeqsolve(
                        terms=term,
                        solver=solver.Dopri5(),
                        t0=0,
                        t1=(num_steps * step_size),
                        dt0=step_size,
                        y0=initial_conditions,
                        args={"geomGrad": geom_grad / 10},
                    )
                    # Extract the deformed source from the integrated states
                    deformed_source = states.ys[0, :, :2]

                    geom_grad = getSinkData(deformed_source)

                    loss, gradients = value_and_grad(lddmm_hamiltonian)(
                        source,
                        target,
                        myAlpha,
                        initial_momentum,
                        num_steps=num_steps,
                        step_size=step_size,
                        geom_grad=geom_grad,
                    )

                    updates, state = optimizer.update(gradients, state)

                    initial_momentum = optax.apply_updates(initial_momentum, updates)

                    MSE_defSource_target.append(float(jnp.mean((deformed_source - target) ** 2)))
                    regLoss.append(float(myAlpha * jnp.mean(initial_momentum**2)))

                    # store innerLoopInfo
                    dataObj.results[f"{testNum}"] = {"deformed_source": deformed_source}
                    dataObj.set_instanceData_toDataFile()
                # store outerLoopInfo
                dataObj.results[f"{testNum}"]["endTime"] = time.strftime("%H:%M:%S", time.localtime())
                dataObj.results[f"{testNum}"]["startTime"] = startTime
                dataObj.results[f"{testNum}"]["MSE_defSource_target"] = np.array(MSE_defSource_target)
                dataObj.results[f"{testNum}"]["regLoss"] = np.array(regLoss)

                testNum += 1

    # dataObj.set_globalData_toDataFile()
    return (dataObj.results, testNum)


##########

# ToDo: turn errorHandling into an ErrorHandler class!
# def errorHandling(self, input, errorGenus, expectedInput=None, metaInfo=None):
#     self.errorHandling(input=input, errorGenus='ValueMissingError', expectedInput='errorHandling(->input)')
#     self.errorHandling(input=errorGenus, errorGenus='ValueMissingError', expectedInput='errorHandling(->errorGenus)')
#     match errorGenus:
#         #must include expectedInfo, as the expected type which the input type is being compared to
#         case 'TypeError':
#             self.errorHandling(input=expectedInput, errorGenus='ValueMissingError', expectedInput='errorHandling(->expectedInput)')
#             if (type(expectedInput).__name__ != 'list') is True:
#                 expectedInput = [expectedInput]
#             if (type(input).__name__ not in expectedInput) is True:
#                 raise TypeError(f'{expectedInput} expected. {type(input)} passed-in instead.')

#         case 'ValueMissingError':
#             if (input is None) is True:
#                 raise ValueMissingError(f'An input was expected for {expectedInput} but None was passed-in.')

#         #must include metaInfo (the class variable's name which is being checked to be empty or not)
#         case 'RealClassVariableEmptyError':
#             self.errorHandling(input=metaInfo, errorGenus='ValueMissingError', expectedInput='errorHandling(->metaInfo)')
#             self.errorHandling(input=metaInfo, errorGenus='TypeError', expectedInput='str')
#             if (len(input)==0) is True:
#                 raise ValueMissingError(f'Attempted to fetch empty class variable {metaInfo}. Set {metaInfo} first.')
