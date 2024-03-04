"""FIX"""

import matplotlib.pyplot as plt
import numpy as np


def slicedDataMesh(slicedData_mesh):
    """FIX"""
    for sliceKey in slicedData_mesh.keys():
        numOfSlices = len(slicedData_mesh[list(slicedData_mesh.keys())[0]])
        numOfSectPerAxis = int(np.sqrt(numOfSlices))
        fig, ax = plt.subplots(nrows=numOfSectPerAxis, ncols=numOfSectPerAxis)
        meshSect = numOfSectPerAxis - 1
        keyToSubplotIdx = {}

        for i in range(numOfSectPerAxis**2):
            keyToSubplotIdx[i] = [int((i - i % numOfSectPerAxis) / numOfSectPerAxis), i % numOfSectPerAxis]


        goNext = lambda x, n=numOfSectPerAxis: x if (x) < n**2 else (x - 1) % n

        n = numOfSectPerAxis

        cmap = plt.colormaps["tab10"]
        colorIdx = np.random.uniform(size=(numOfSectPerAxis**2))
        i = 0
        for thisSect in range(numOfSectPerAxis**2):
            key = tuple(keyToSubplotIdx[thisSect])

            if slicedData_mesh[sliceKey][f"{meshSect}"].shape[0] == 0:
                ax[key].scatter(
                    slicedData_mesh[sliceKey][f"{meshSect}"][:],
                    slicedData_mesh[sliceKey][f"{meshSect}"][:],
                    s=0.005,
                    label=f"Q{meshSect}",
                    color=cmap(colorIdx[i]),
                )

            else:
                ax[key].scatter(
                    slicedData_mesh[sliceKey][f"{meshSect}"][:, 0],
                    slicedData_mesh[sliceKey][f"{meshSect}"][:, 1],
                    s=0.005,
                    label=f"Q{meshSect}",
                    color=cmap(colorIdx[i]),
                )
            meshSect = goNext(meshSect + n)
            ax[key].axis("off")
        i += 1
        fig.suptitle(f"Slice: {sliceKey}")
        plt.show()


"""
  processedData['source_subSampled'] = source_downSampled
    processedData['target'] = np.concatenate([slicedData_mesh[targetLabel][f'{slice}'] for slice in slicedData_mesh[targetLabel]])
    processedData['source'] = np.concatenate([slicedData_mesh[sourceLabel][f'{slice}'] for slice in slicedData_mesh[sourceLabel]])
    processedData['metaData'] = {'sourceLabel': sourceLabel,
                                 'targetLabel': targetLabel}
"""


def downSampled(processedData):
    """FIX"""
    target = processedData["target"]
    source = processedData["source"]
    source_subSamplede = processedData["source_subSampled"]

    targetLabel = processedData["metaData"]["targetLabel"]
    sourceLabel = processedData["metaData"]["sourceLabel"]

    cmap = plt.colormaps["tab10"]
    colorIdx = np.random.uniform(size=6)

    plt.scatter(target[:, 0], target[:, 1], marker="x", color=cmap(colorIdx[0]), zorder=3, s=0.15, alpha=0.4)
    plt.suptitle(f"Target\n{targetLabel}")
    plt.show()
    plt.scatter(source[:, 0], source[:, 1], marker="x", color=cmap(colorIdx[2]), zorder=3, s=0.15, alpha=0.4)
    plt.suptitle(f"Source\n{sourceLabel}")
    plt.show()
    plt.scatter(
        source_subSamplede[:, 0],
        source_subSamplede[:, 1],
        marker="x",
        color=cmap(colorIdx[2]),
        zorder=3,
        s=0.15,
        alpha=0.4,
    )
    plt.suptitle(f"Source_SubSampled\n{sourceLabel}")
    plt.show()


"""
axis alpha/rank/numOfIterations/
"""


def resultsPlotter(resultsData, axis=None):
    """FIX"""
    return 0
