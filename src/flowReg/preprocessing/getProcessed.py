# from meshData, interested in sliceLabel_1 and sliceLabel_2
# choose smallest as target, and largest as source -> downSample the source to match the target
"""FIX"""
import numpy as np

"""

input : meshedData (dataDict), sliceLabel_1 (str), sliceLabel_2 (str), metDataToggle = False (bool)

returns processedData dict such that
    processedData['source_subSampled'] <- subSampledSource
    processedData['target'] <- target (smaller of the two inputs)
    processedData['metaData']['sourceLabel'] <- sourceLabel from the two input labels
    processedData['metaData']['targetLabel'] <- targetLabel from the two input labels

    if metaDataToggle == True:
         processedData['metaData'] <- extraInfo[meshSection_key] from spaghetti


"""


def downSampled(slicedData_mesh, sliceLabel_1, sliceLabel_2, metaDataToggle=False):
    """FIX"""
    extraInfo = {}
    processedData = {}

    sourceLabel = sliceLabel_1
    targetLabel = sliceLabel_2

    sourceLen = sum(len(slicedData_mesh[sourceLabel][f"{slice}"]) for slice in slicedData_mesh[sourceLabel])
    targetLen = sum(len(slicedData_mesh[targetLabel][f"{slice}"]) for slice in slicedData_mesh[targetLabel])

    if sourceLen < targetLen:
        sourceLabel = sliceLabel_2
        targetLabel = sliceLabel_1
        tempLen = sourceLen
        sourceLen = targetLen
        targetLen = tempLen

    leaveOutNum = sourceLen - targetLen
    leaveOutRatio = leaveOutNum / sourceLen
    countPoints = 0
    numOfPtsMissing = 0

    source_downSampled = []
    leftOutDataPts = []
    for meshSection_key in slicedData_mesh[targetLabel].keys():
        source_currSect = np.copy(slicedData_mesh[sourceLabel][meshSection_key])
        sourceLen_currSect = len(source_currSect)

        numSamplesToCut = int(np.round(sourceLen_currSect * leaveOutRatio, 0))
        numSamplesToKeep = sourceLen_currSect - numSamplesToCut

        if numSamplesToCut == 0:
            numSamplesToKeep = 0

        else:
            countPoints += numSamplesToKeep
            indicesToKeep = np.random.randint(sourceLen_currSect, size=numSamplesToKeep)
            indicesToCut = np.delete(np.arange(len(source_currSect)), indicesToKeep)
            # source_downSampled += [x for x in (source_currSect[indicesToKeep, :])]
            # leftOutDataPts += [x for x in (source_currSect[indicesToCut, :])]
            source_downSampled += list(source_currSect[indicesToKeep, :])
            leftOutDataPts += list(source_currSect[indicesToCut, :])
        try:
            percentSourceUsed = (sourceLen_currSect - numSamplesToCut) / sourceLen_currSect
        except ZeroDivisionError:
            percentSourceUsed = 0
        if metaDataToggle is True:
            extraInfo[meshSection_key] = {
                "numOfTotalSourcePoints": sourceLen_currSect,
                "sourcePointsUsed": numSamplesToKeep,
                "percentSourcePointsUsed": percentSourceUsed,
                "sourcePointsLeftOut": (sourceLen_currSect - numSamplesToKeep),
                "percentSourcePointsLeftOut": (1 - percentSourceUsed),
            }
    numOfPtsMissing = (sourceLen - leaveOutNum) - countPoints
    # countMakeupPoints = 0

    # for meshSection_key in slicedData_mesh[targetLabel].keys():
    #     source_currSect = slicedData_mesh[sourceLabel][meshSection_key]
    #     for x in source_currSect:
    #         if x not in np.array(source_downSampled):
    #             source_downSampled += [x]
    #             countMakeupPoints += 1
    #             if metaDataToggle == True:
    #                 extraInfo[meshSection_key]['sourcePointsUsed'] += 1
    #                 extraInfo[meshSection_key]['sourcePointsLeftOut'] -= 1
    #                 extraInfo[meshSection_key]['percentSourcePointsUsed'] =\
    #                     (extraInfo[meshSection_key]['numOfTotalSourcePoints']-\
    #                      extraInfo[meshSection_key]['sourcePointsUsed'])/\
    #                      extraInfo[meshSection_key]['numOfTotalSourcePoints']
    #                 extraInfo[meshSection_key]['percentSourcePointsLeftOut'] = 1 - extraInfo[meshSection_key]['percentSourcePointsUsed']
    #         if countMakeupPoints == numOfPtsMissing:
    #             break
    #     if countMakeupPoints == numOfPtsMissing:
    #         break
    leftOutDataPts = np.array(leftOutDataPts)
    indiciesToAddBackIn = np.random.randint(len(leftOutDataPts), size=numOfPtsMissing)
    # source_downSampled += [x for x in leftOutDataPts[indiciesToAddBackIn, :]]
    source_downSampled += list(leftOutDataPts[indiciesToAddBackIn, :])
    source_downSampled = np.array(source_downSampled, copy=True)

    processedData["source_subSampled"] = source_downSampled
    processedData["target"] = np.concatenate(
        [slicedData_mesh[targetLabel][f"{slice}"] for slice in slicedData_mesh[targetLabel]]
    )
    processedData["source"] = np.concatenate(
        [slicedData_mesh[sourceLabel][f"{slice}"] for slice in slicedData_mesh[sourceLabel]]
    )
    processedData["metaData"] = {"sourceLabel": sourceLabel, "targetLabel": targetLabel}
    if metaDataToggle is True:
        processedData["metaData"]["downSamplingDetails"] = extraInfo
    return processedData
