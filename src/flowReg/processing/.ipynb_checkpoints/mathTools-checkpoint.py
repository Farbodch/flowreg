"""FIX"""
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from diffrax import ODETerm, diffeqsolve, solver
from jax import value_and_grad, jit
from functools import partial
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
    """A class to store and manage data throughout the computation pipeline.
    This class also keeps track of the format friendly to the plotter function
    (.tools.plotter), as well as writing to (TO DO), and reading from (TO DO), file.

    Parameters
    ----------
    results
        Dictionary object: to store the results of the matching computation.
    source
        Numpy array: to store source's spatial data.
    target
        Numpy array: to store target's spatial data.
    source_label
        String: storing label of the source (decorative, not functional).
    target_label
        String: storing label of the target (decorative, not functional).
    alpha_list
        List of floats: regularization strengths to assess the loss at.
    rank_list
        List of integers: ranks at which the low-rank sinkhorn divergence
        algorithm is assessed at.
    iterations_list
        List of integers: number of iterations the optimization algorithm should
        loop through.

    """

    results = {}
    def __init__(
        self,
        source=np.array([]),
        target=np.array([]),
        source_label="",
        target_label="",
        alpha_list=None,
        iterations_list=None,
        rank_list=None,
    ):

        if alpha_list is None:
            alpha_list = []
        if iterations_list is None:
            iterations_list = []
        if rank_list is None:
            rank_list = []

        self.source = source
        self.target = target
        self.source_label = source_label
        self.target_label = target_label
        self.alpha_list = alpha_list
        self.iterations_list = iterations_list
        self.rank_list = rank_list
        self.reset_instanceData()
        self.test_num = 0

    def reset_instanceData(self):
        """Resetting 'private' parameters of the class. These are used during outer
        loop of the computation pipeline, when assessing different parameters passed
        in through alpha_list, rank_list, and iterations_list.

        Parameters
        ----------
        N/A

        Returns
        -------
        N/A
        """
        self.MSE_defSource_target_ = None
        self.regLoss_ = None
        self.startTime_ = None
        self.deformed_source_ = None

        self.alpha_ = None
        self.iterations_ = None
        self.rank_ = None
        self.num_of_steps_ = None
        self.step_size_ = None

    # check to ensure not overriding an existing self._test_num!
    def set_instanceData_toDataFile(self):
        """Saves the parameter data of the current instance being
        computed (as indexed by test_num) to the results dictionary.

        Parameters
        ----------
        N/A

        Returns
        -------
        N/A
        """
        self.results[f"{self.test_num}"]["instanceMetaData"] = {
            "alpha": self.alpha_,
            "numOfOptimItrs": self.iterations_,
            "lr_sink_rank": self.rank_,
            "integratorNumSteps": self.num_of_steps_,
            "integratorStepSize": self.step_size_,
        }


def vector_field(t, state, params):
    """Generate the time derivative (vector field) of ODE's current state.

    Parameters
    ----------
    t
        A float: required for the Diffrax ode package's bookkeeping.
    state
        A Tensor: that keeps track of current state of the ODE's
        current state (current position and momentum of each data
        point along all the relevant spatial directions).
    params
        A Dictionary: that keeps track of static passed-in instaces
        that are not affected by the ODE evolution, but influence
        the evolution. For us, it passes in the geometric gradient
        of our points in space <!--Testing:, as well as the function that can
        compute this geometric gradient-->.

    Returns
    -------
    A Tensor containing the vector field values with regards to both position
    (dX/dt) and momentum (dP/dt), of each individual data point (along
    all relevant dimensions/axes).
    """
    # V=dX/dT, dP/dT=m.dV/dT=m.V2-V1, let m=1, dT=1
    X, P = jnp.split(state, 2, axis=-1)

    # geomOTCostFen = params["geomOTCostFen"]
    # geomGradCurr = geomOTCostFen(X)
    # V = P + geomGradCurr ###Testomg - curr
    V = P + params["geomGrad"] ###Testing - Prev

    return jnp.concatenate([V, -V], axis=-1)

def evolve_points(source, momentum, geom_grad, geomOTCostFen, num_steps=10, step_size=0.1):
    """Evolves the source points passed in, by integrating and pushing them along a vector
    field defined by their geometric gradient.

    Parameters
    ----------
    source
        Numpy array: of spatial coordinates of data points.
    momentum
        Numpy array: of the momentum value of data points along relevant dimensions/axes.
    geom_grad
        Numpy array: geometric gradient of the passed in source points.

    <!--Testing:
    geomOTCostFen
        Function: that calculates and returns geometric gradient values -->

    num_steps
        An integer: indicating the number of steps our discretized ODE is going to be
        integrated along.
    step_size
        A float: indicating the size step size of our discretized ODE's integrator's
        step size.

    Returns
    -------
    A numpy array that of the new spatial coordinates of our source points after being evolved
    through the ODE.
    """
    # The state consists of [X, P] where X is the point cloud and P is its momentum
    initial_conditions = jnp.concatenate([source, momentum], axis=-1)

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
        args={"geomGrad": geom_grad, "geomOTCostFen": geomOTCostFen},
    )

    # Extract the deformed source from the integrated states
    # states.ys has shape 1*numOfPts*4 where 0*x*0:2 is the position of each point & 0*x*2:4 is the momentum of each point
    deformed_source = states.ys[0, :, :2]

    return deformed_source

def lddmm_objective_function(source,
                             target,
                             alpha,
                             momentum,
                             geom_grad,
                             geomOTCostFen,
                             num_steps=10,
                             step_size=0.1):
    """The objective function of our optimization problem. This function returns the regularized
    loss between our evolved source points, and our target.

    Parameters
    ----------
    source
        Numpy array: of spatial coordinates of data points we're attemping
        to evolve and match with target.
    target
        Numpy array: of spatial coordinates of data points we're attemping
        to match the source points with.
    alpha
        A float: regularization strength of our optimization problem.
    momentum
        Numpy array: of the momentum value of data points along relevant dimensions/axes.
    geom_grad
        Numpy array: geometric gradient of the passed in source points.

    <!--Testing:
    geomOTCostFen
        Function: that calculates and returns geometric gradient values -->

    num_steps
        An integer: indicating the number of steps our discretized ODE is going to be
        integrated along.
    step_size
        A float: indicating the size step size of our discretized ODE's integrator's
        step size.

    Returns
    -------
    N/A
    """

    deformed_source = evolve_points(source, momentum, geom_grad, geomOTCostFen, num_steps, step_size)
    # Compute the L2 loss between the deformed source and target
    loss = (jnp.mean(jnp.square(deformed_source - target)) + alpha * jnp.mean(jnp.square(momentum))) ** (1 / 1)

    return loss


def sink_lr_setRank(curr_rank):
    """FIX"""

    def sink_lr_cost(geom):
        """Return the OT cost and OT output given a geometry"""
        ot = sinkhorn_lr.LRSinkhorn(rank=curr_rank, initializer="random")(linear_problem.LinearProblem(geom))
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

        # assert ot.converged


        # weights = ot.geoms[0].transport_from_potentials(ot.potentials[0][0],ot.potentials[0][1])
        # ots[10].geoms[0].transport_from_potentials(ots[10].potentials[0][0],ots[10].potentials[0][1])
        # transportMatrix = ot.geoms[0].transport_from_potentials(ot.potentials[0][0],ot.potentials[0][1])
        return geom_g.x

    return getWeightsInner


def runMatching(dataObj, num_steps=10, step_size=0.1):
    """FIX"""
    dataObj.reset_instanceData()

    alpha_list = dataObj.alpha_list
    rank_list = dataObj.rank_list
    iterations_list = dataObj.iterations_list

    source = dataObj.source
    target = dataObj.target

    dataObj.num_of_steps_ = num_steps
    dataObj.step_size_ = step_size

    test_num = dataObj.test_num

    # OuterForLoop Iterating over params
    for curr_alpha in alpha_list:
        for curr_rank in rank_list:
            sink_lr_getCost = sink_lr_setRank(curr_rank)
            getSinkData = getWeightsSetTarget(target, cost_fen=sink_lr_getCost)
            for curr_num_of_itrs in iterations_list:
                dataObj.test_num = test_num

                dataObj.alpha_ = curr_alpha
                dataObj.rank_ = curr_rank
                dataObj.iterations_ = curr_num_of_itrs

                optimizer = optax.adam(1e-1)

                momentum = jnp.zeros_like(source)
                state = optimizer.init(momentum)

                MSE_defSource_target = []
                regLoss = []

                startTime = time.strftime("%H:%M:%S", time.localtime())
                for _i in range(curr_num_of_itrs):

                    geom_grad = getSinkData(source)

                    deformed_source = evolve_points(
                        source,
                        momentum,
                        num_steps=num_steps,
                        step_size=step_size,
                        geom_grad=geom_grad,
                        geomOTCostFen=getSinkData
                    )

                    loss, gradients = value_and_grad(lddmm_objective_function)(
                        source,
                        target,
                        curr_alpha,
                        momentum,
                        num_steps=num_steps,
                        step_size=step_size,
                        geom_grad=geom_grad,
                        geomOTCostFen=getSinkData
                    )

                    updates, state = optimizer.update(gradients, state)

                    momentum = optax.apply_updates(momentum, updates)

                    MSE_defSource_target.append(float(jnp.mean((deformed_source - target) ** 2)))
                    regLoss.append(float(curr_alpha * jnp.mean(momentum**2)))

                    # store innerLoopInfo
                    dataObj.results[f"{test_num}"] = {"deformed_source": deformed_source}
                    dataObj.set_instanceData_toDataFile()
                # store outerLoopInfo
                dataObj.results[f"{test_num}"]["endTime"] = time.strftime("%H:%M:%S", time.localtime())
                dataObj.results[f"{test_num}"]["startTime"] = startTime
                dataObj.results[f"{test_num}"]["MSE_defSource_target"] = np.array(MSE_defSource_target)
                dataObj.results[f"{test_num}"]["regLoss"] = np.array(regLoss)

                test_num += 1

    # dataObj.set_globalData_toDataFile()
    return (dataObj.results, test_num)


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
