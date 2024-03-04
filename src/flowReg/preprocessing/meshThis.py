"""FIX"""

import numpy as np


# struct:
# data + meshResolution -> getDataMesh
# data -> getSlicedData ->> slicedData
# meshResolution + slicedData -> getSlicedData_gridBins:
#                                      gridResolution(meshResolution)+ slicedData -> getDataGrid:
#                                            slicedData -> getDataBounds --> dataBounds
#                                                                                --> dataGrid
#                            --> slicedData_gridBins
# --> slicedData_mesh
def getDataMesh(data, meshResolution):
    """FIX"""
    slicedData = getSlicedData(data)
    slicedData_gridBins = getSlicedData_gridBins(slicedData, meshResolution)
    slicedData_mesh = {}
    for currSlice in slicedData.keys():
        slicedData_mesh[currSlice] = {}
        numOfMeshSections = len(slicedData_gridBins[currSlice])
        for meshSectIdx in range(numOfMeshSections):
            slicedData_mesh[currSlice][f"{meshSectIdx}"] = []
            # slicedData_mesh[currSlice][meshSectIdx] = []

        for dataPt in slicedData[currSlice]:
            toThisBin = assignBin(dataPt, slicedData_gridBins[currSlice])
            slicedData_mesh[currSlice][f"{toThisBin}"] += [dataPt]

        for thisBin in slicedData_mesh[currSlice].keys():
            slicedData_mesh[currSlice][thisBin] = np.array(slicedData_mesh[currSlice][thisBin], copy=True)

    return slicedData_mesh


def getSlicedData(data):
    """FIX"""
    spatial_coords_aff = data.obsm["spatial_norm_affine"]
    dataLen = len(data.obs["batch"])
    slicedDataAnn = data.obs["batch"].unique().tolist()
    annIdx = data.obs["batch"].tolist()
    # slicedDataAnn
    slicedData = {}

    for categoryIdx in range(len(slicedDataAnn)):
        slicedData[slicedDataAnn[categoryIdx]] = []
    for i in range(dataLen):
        slicedData[annIdx[i]] += [spatial_coords_aff[i].tolist()]
    for key in slicedData.keys():
        slicedData[key] = np.array(slicedData[key])
    return slicedData


def getSlicedData_gridBins(slicedData, gridResolution=2):
    """FIX"""
    dataGrid = getDataGrid(slicedData, gridResolution)
    myGridRes = gridResolution
    slicedData_gridBins = {}
    for currSlice in dataGrid.keys():
        # numOfAxes = len(dataGrid[list(dataGrid.keys())[0]])
        # maybe a generalized outer loop for higher than 2D tensoring/meshing?
        # Below only works for 2D Data!! Can't mesh anything higher! Need to restructure below for >2D meshes
        meshThis = 0
        currGridBin = []
        while meshThis < myGridRes:
            for meshThat in range(myGridRes):
                thisXBound = dataGrid[currSlice][0][meshThis]
                thisYBound = dataGrid[currSlice][1][meshThat]
                currGridBin += [[thisXBound, thisYBound]]
            meshThis = meshThis + 1
        slicedData_gridBins[currSlice] = np.array(currGridBin)
    return slicedData_gridBins


def getDataGrid(slicedData, gridResolution):
    """FIX"""
    dataBounds = getDataBounds(slicedData)
    dataGrid = {}
    myGridRes = gridResolution
    for currSlice in dataBounds.keys():
        dataGrid[currSlice] = getGridDoms(dataBounds[currSlice], myGridRes)
    return dataGrid


def getDataBounds(slicedData):
    """FIX"""
    dataBounds = {}
    for currSlice in slicedData.keys():
        numOfAxes = len(slicedData[currSlice][0])
        currSliceBounds = []
        for currAxis in range(numOfAxes):
            minCurrAxis = np.round(min(slicedData[currSlice][:, currAxis] - 0.175), 1)
            maxCurrAxis = np.round(max(slicedData[currSlice][:, currAxis] + 0.175), 1)
            currSliceBounds += [[minCurrAxis, maxCurrAxis]]
        dataBounds[currSlice] = currSliceBounds
    return dataBounds


def getGridDoms(domains, gridRes):
    """FIX"""
    grid = []
    for x_i in range(len(domains)):
        grid_i = []
        l_bound = np.min(domains[x_i])
        L = np.max(domains[x_i]) - np.min(domains[x_i])
        dx = L / gridRes
        for _i in range(gridRes):
            grid_i.append([np.round(l_bound, 1), np.round(l_bound + dx, 1)])
            l_bound = l_bound + dx
        grid += [grid_i]
    return grid


def assignBin(dataPoint, gridBins):
    # symmetric mesh so sqrt of the num of bins will always be an integer
    """FIX"""
    gridRes = int(np.sqrt(len(gridBins)))
    for thisBinIdx in range(len(gridBins)):
        xCheck = False
        yCheck = False
        # for currAxis in range(len(dataPoint)):
        axis_0_Idx = 0
        axis_1_Idx = 1
        minBoundIdx = 0
        maxBoundIdx = 1

        # x-axis check:
        # we're at the right-boundary of the 2D data space/mesh
        if thisBinIdx >= (gridRes**2 - gridRes):
            if (
                dataPoint[axis_0_Idx] >= gridBins[thisBinIdx][axis_0_Idx][minBoundIdx]
                and dataPoint[axis_0_Idx] <= gridBins[thisBinIdx][axis_0_Idx][maxBoundIdx]
            ):
                xCheck = True
        # everywhere else
        else:
            if (
                dataPoint[axis_0_Idx] >= gridBins[thisBinIdx][axis_0_Idx][minBoundIdx]
                and dataPoint[axis_0_Idx] < gridBins[thisBinIdx][axis_0_Idx][maxBoundIdx]
            ):
                xCheck = True

        # y-axis check:
        # we're at the top-boundary of the 2D data space/mesh
        if (thisBinIdx + 1) % gridRes == 0:
            if (
                dataPoint[axis_1_Idx] >= gridBins[thisBinIdx][axis_1_Idx][minBoundIdx]
                and dataPoint[axis_1_Idx] <= gridBins[thisBinIdx][axis_1_Idx][maxBoundIdx]
            ):
                yCheck = True
        else:
            if (
                dataPoint[axis_1_Idx] >= gridBins[thisBinIdx][axis_1_Idx][minBoundIdx]
                and dataPoint[axis_1_Idx] < gridBins[thisBinIdx][axis_1_Idx][maxBoundIdx]
            ):
                yCheck = True

        if xCheck and yCheck:
            return thisBinIdx

    return -1
