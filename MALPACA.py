##BASE PYTHON
import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import glob
import copy
import multiprocessing
import vtk.util.numpy_support as vtk_np
import numpy as np

#
# MALPACA
#

class MALPACA(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
  
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "MALPACA" # TODO make this more human readable by adding spaces
    self.parent.categories = ["SlicerMorph.In Development"]
    self.parent.dependencies = []
    self.parent.contributors = ["Arthur Porto, Sara Rolfe (UW), Murat Maga (UW)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
      This module automatically transfers landmarks on a reference 3D model (mesh) to a target 3D model using dense correspondence and deformable registration. First optimize the parameters in single alignment analysis, then use them in batch mode to apply to all 3D models
      """
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
      This module was developed by Arthur Porto, Sara Rolfe, and Murat Maga, through a NSF ABI Development grant, "An Integrated Platform for Retrieval, Visualization and Analysis of
      3D Morphology From Digital Biological Collections" (Award Numbers: 1759883 (Murat Maga), 1759637 (Adam Summers), 1759839 (Douglas Boyer)).
      https://nsf.gov/awardsearch/showAward?AWD_ID=1759883&HistoricalAwards=false
      """ # replace with organization, grant and thanks.      

#
# MALPACAWidget
#

class MALPACAWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    try:
      import open3d as o3d
      print('o3d installed')
    except ModuleNotFoundError as e:
      if slicer.util.confirmOkCancelDisplay("MALPACA requires the open3d library. Installation may take a few minutes"):
        slicer.util.pip_install('notebook==6.0.3')
        slicer.util.pip_install('open3d==0.9.0')
        import open3d as o3d
    try:
      from pycpd import DeformableRegistration
      print('pycpd installed')
    except ModuleNotFoundError as e:
      slicer.util.pip_install('git+https://github.com/agporto/pycpd.git@development')
      print('trying to install pycpd')
      from pycpd import DeformableRegistration
    
  def onSelect(self):
    self.applyButton.enabled = bool (self.meshDirectory.currentPath and self.landmarkDirectory.currentPath and 
      self.sourceModelSelector.currentNode() and self.baseLMSelector.currentNode() and self.baseSLMSelect.currentNode() 
      and self.outputDirectory.currentPath)
          
  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    
    # Set up tabs to split workflow
    tabsWidget = qt.QTabWidget()
    alignSingleTab = qt.QWidget()
    alignSingleTabLayout = qt.QFormLayout(alignSingleTab)

    tabsWidget.addTab(alignSingleTab, "Multi-Template Alignment")
    self.layout.addWidget(tabsWidget)
    
    # Layout within the tab
    alignSingleWidget=ctk.ctkCollapsibleButton()
    alignSingleWidgetLayout = qt.QFormLayout(alignSingleWidget)
    alignSingleWidget.text = "Align target meshes to multiple templates "
    alignSingleTabLayout.addRow(alignSingleWidget)
  
    #
    # Select source mesh directory
    #
    self.sourceModelSelector = ctk.ctkPathLineEdit()
    self.sourceModelSelector.filters  = ctk.ctkPathLineEdit.Dirs
    alignSingleWidgetLayout.addRow("Source meshes directory: ", self.sourceModelSelector)
    
    #
    # Select source landmark directory
    #
    self.sourceFiducialSelector = ctk.ctkPathLineEdit()
    self.sourceFiducialSelector.filters  = ctk.ctkPathLineEdit.Dirs
    alignSingleWidgetLayout.addRow("Source landmarks directory: ", self.sourceFiducialSelector)
    
    #
    # Select target mesh directory
    #
    self.targetModelSelector = ctk.ctkPathLineEdit()
    self.targetModelSelector.filters  = ctk.ctkPathLineEdit.Dirs
    alignSingleWidgetLayout.addRow("Target meshes directory: ", self.targetModelSelector)

    #
    # Select output directory
    #
    self.landmarkOutputSelector = ctk.ctkPathLineEdit()
    self.landmarkOutputSelector.filters  = ctk.ctkPathLineEdit.Dirs
    alignSingleWidgetLayout.addRow("Output directory: ", self.landmarkOutputSelector)

    #
    # Select whether to skip scaling during alignment
    #
    self.skipScalingCheckBox = qt.QCheckBox()
    self.skipScalingCheckBox.checked = 0
    self.skipScalingCheckBox.setToolTip("If checked, MALPACA will skip scaling during the alignment (Not recommended).")
    alignSingleWidgetLayout.addRow("Skip scaling", self.skipScalingCheckBox)

    #
    # Select whether to skip the final projection step
    #    
    self.skipProjectionCheckBox = qt.QCheckBox()
    self.skipProjectionCheckBox.checked = 0
    self.skipProjectionCheckBox.setToolTip("If checked, MALPACA will skip final refinement step placing landmarks on the target suface.")
    alignSingleWidgetLayout.addRow("Skip projection", self.skipProjectionCheckBox)
    

    [self.projectionFactor,self.pointDensity, self.normalSearchRadius, self.FPFHSearchRadius, self.distanceThreshold, self.maxRANSAC, self.maxRANSACValidation, 
    self.ICPDistanceThreshold, self.alpha, self.beta, self.CPDIterations, self.CPDTolerence] = self.addAdvancedMenu(alignSingleWidgetLayout)
    
    # Advanced tab connections
    self.projectionFactor.connect('valueChanged(double)', self.onChangeAdvanced)
    self.pointDensity.connect('valueChanged(double)', self.onChangeAdvanced)
    self.normalSearchRadius.connect('valueChanged(double)', self.onChangeAdvanced)
    self.FPFHSearchRadius.connect('valueChanged(double)', self.onChangeAdvanced)
    self.distanceThreshold.connect('valueChanged(double)', self.onChangeAdvanced)
    self.maxRANSAC.connect('valueChanged(double)', self.onChangeAdvanced)
    self.maxRANSACValidation.connect('valueChanged(double)', self.onChangeAdvanced)
    self.ICPDistanceThreshold.connect('valueChanged(double)', self.onChangeAdvanced)
    self.alpha.connect('valueChanged(double)', self.onChangeAdvanced)
    self.beta.connect('valueChanged(double)', self.onChangeAdvanced)
    self.CPDIterations.connect('valueChanged(double)', self.onChangeAdvanced)
    self.CPDTolerence.connect('valueChanged(double)', self.onChangeAdvanced)
    
    #
    # Run Button
    #
    self.runButton = qt.QPushButton("Run auto-landmarking")
    self.runButton.toolTip = "Align the target meshes with multiple templates and transfer landmarks."
    self.runButton.enabled = False
    alignSingleWidgetLayout.addRow(self.runButton)
    
    
    # connections
    self.sourceModelSelector.connect('validInputChanged(bool)', self.onSelect)
    self.sourceFiducialSelector.connect('validInputChanged(bool)', self.onSelect)
    self.targetModelSelector.connect('validInputChanged(bool)', self.onSelect)
    self.landmarkOutputSelector.connect('validInputChanged(bool)', self.onSelect)
    self.runButton.connect('clicked(bool)', self.onRun)
    
    # initialize the parameter dictionary from single run parameters
    self.parameterDictionary = {
      "projectionFactor": self.projectionFactor.value,
      "pointDensity": self.pointDensity.value,
      "normalSearchRadius" : self.normalSearchRadius.value,
      "FPFHSearchRadius" : self.FPFHSearchRadius.value,
      "distanceThreshold" : self.distanceThreshold.value,
      "maxRANSAC" : int(self.maxRANSAC.value),
      "maxRANSACValidation" : int(self.maxRANSACValidation.value),
      "ICPDistanceThreshold"  : self.ICPDistanceThreshold.value,
      "alpha" : self.alpha.value,
      "beta" : self.beta.value,
      "CPDIterations" : int(self.CPDIterations.value),
      "CPDTolerence" : self.CPDTolerence.value
      }
  
  def cleanup(self):
    pass
  
  def onSelect(self):
    self.runButton.enabled = bool ( self.sourceModelSelector.currentPath and self.targetModelSelector.currentPath and self.sourceFiducialSelector.currentPath and self.landmarkOutputSelector.currentPath)
    
    
  def updateLayout(self):
    layoutManager = slicer.app.layoutManager()
    layoutManager.setLayout(9)  #set layout to 3D only
    layoutManager.threeDWidget(0).threeDView().resetFocalPoint()
    layoutManager.threeDWidget(0).threeDView().resetCamera()
    
  def onChangeAdvanced(self):
    self.updateParameterDictionary()

  def onRun(self):
    logic = MALPACALogic()
    if self.skipProjectionCheckBox.checked != 0:
      projectionFactor = 0
    else:  
      projectionFactor = self.projectionFactor.value/100
      
    logic.runLandmarkMultiprocess(self.sourceModelSelector.currentPath,self.sourceFiducialSelector.currentPath, 
    self.targetModelSelector.currentPath, self.landmarkOutputSelector.currentPath, self.skipScalingCheckBox.checked, projectionFactor,self.parameterDictionary)
    
    
  def updateParameterDictionary(self):    
    # update the parameter dictionary from single run parameters
    if hasattr(self, 'parameterDictionary'):
      self.parameterDictionary["projectionFactor"] = self.projectionFactor.value
      self.parameterDictionary["pointDensity"] = self.pointDensity.value
      self.parameterDictionary["normalSearchRadius"] = int(self.normalSearchRadius.value)
      self.parameterDictionary["FPFHSearchRadius"] = int(self.FPFHSearchRadius.value)
      self.parameterDictionary["distanceThreshold"] = self.distanceThreshold.value
      self.parameterDictionary["maxRANSAC"] = int(self.maxRANSAC.value)
      self.parameterDictionary["maxRANSACValidation"] = int(self.maxRANSACValidation.value)
      self.parameterDictionary["ICPDistanceThreshold"] = self.ICPDistanceThreshold.value
      self.parameterDictionary["alpha"] = self.alpha.value
      self.parameterDictionary["beta"] = self.beta.value
      self.parameterDictionary["CPDIterations"] = int(self.CPDIterations.value)
      self.parameterDictionary["CPDTolerence"] = self.CPDTolerence.value
    

      
  def addAdvancedMenu(self, currentWidgetLayout):
    #
    # Advanced menu for single run
    #
    advancedCollapsibleButton = ctk.ctkCollapsibleButton()
    advancedCollapsibleButton.text = "Advanced parameter settings"
    advancedCollapsibleButton.collapsed = True
    currentWidgetLayout.addRow(advancedCollapsibleButton)
    advancedFormLayout = qt.QFormLayout(advancedCollapsibleButton)

    # Point density label
    pointDensityCollapsibleButton=ctk.ctkCollapsibleButton()
    pointDensityCollapsibleButton.text = "Point density and max projection"
    advancedFormLayout.addRow(pointDensityCollapsibleButton)
    pointDensityFormLayout = qt.QFormLayout(pointDensityCollapsibleButton)

    # Rigid registration label
    rigidRegistrationCollapsibleButton=ctk.ctkCollapsibleButton()
    rigidRegistrationCollapsibleButton.text = "Rigid registration"
    advancedFormLayout.addRow(rigidRegistrationCollapsibleButton)
    rigidRegistrationFormLayout = qt.QFormLayout(rigidRegistrationCollapsibleButton)
    
    # Deformable registration label
    deformableRegistrationCollapsibleButton=ctk.ctkCollapsibleButton()
    deformableRegistrationCollapsibleButton.text = "Deformable registration"
    advancedFormLayout.addRow(deformableRegistrationCollapsibleButton)
    deformableRegistrationFormLayout = qt.QFormLayout(deformableRegistrationCollapsibleButton)
    
    # Point Density slider
    pointDensity = ctk.ctkSliderWidget()
    pointDensity.singleStep = 0.1
    pointDensity.minimum = 0.1
    pointDensity.maximum = 3
    pointDensity.value = 1
    pointDensity.setToolTip("Adjust the density of the pointclouds. Larger values increase the number of points, and vice versa.")
    pointDensityFormLayout.addRow("Point Density Adjustment: ", pointDensity)

    # Set max projection factor
    projectionFactor = ctk.ctkSliderWidget()
    projectionFactor.enabled = True
    projectionFactor.singleStep = 1
    projectionFactor.minimum = 0
    projectionFactor.maximum = 10
    projectionFactor.value = 1
    projectionFactor.setToolTip("Set maximum point projection as a percentage of the image diagonal. Point projection is used to make sure predicted landmarks are placed on the target mesh.")
    pointDensityFormLayout.addRow("Maximum projection factor : ", projectionFactor)

    # Normal search radius slider
    
    normalSearchRadius = ctk.ctkSliderWidget()
    normalSearchRadius.singleStep = 1
    normalSearchRadius.minimum = 2
    normalSearchRadius.maximum = 12
    normalSearchRadius.value = 2
    normalSearchRadius.setToolTip("Set size of the neighborhood used when computing normals")
    rigidRegistrationFormLayout.addRow("Normal search radius: ", normalSearchRadius)
    
    #FPFH Search Radius slider
    FPFHSearchRadius = ctk.ctkSliderWidget()
    FPFHSearchRadius.singleStep = 1
    FPFHSearchRadius.minimum = 3
    FPFHSearchRadius.maximum = 20
    FPFHSearchRadius.value = 5
    FPFHSearchRadius.setToolTip("Set size of the neighborhood used when computing FPFH features")
    rigidRegistrationFormLayout.addRow("FPFH Search radius: ", FPFHSearchRadius)
    
    
    # Maximum distance threshold slider
    distanceThreshold = ctk.ctkSliderWidget()
    distanceThreshold.singleStep = .25
    distanceThreshold.minimum = 0.5
    distanceThreshold.maximum = 4
    distanceThreshold.value = 1.5
    distanceThreshold.setToolTip("Maximum correspondence points-pair distance threshold")
    rigidRegistrationFormLayout.addRow("Maximum corresponding point distance: ", distanceThreshold)

    # Maximum RANSAC iterations slider
    maxRANSAC = ctk.ctkDoubleSpinBox()
    maxRANSAC.singleStep = 1
    maxRANSAC.setDecimals(0)
    maxRANSAC.minimum = 1
    maxRANSAC.maximum = 500000000
    maxRANSAC.value = 4000000
    maxRANSAC.setToolTip("Maximum number of iterations of the RANSAC algorithm")
    rigidRegistrationFormLayout.addRow("Maximum RANSAC iterations: ", maxRANSAC)

    # Maximum RANSAC validation steps
    maxRANSACValidation = ctk.ctkDoubleSpinBox()
    maxRANSACValidation.singleStep = 1
    maxRANSACValidation.setDecimals(0)
    maxRANSACValidation.minimum = 1
    maxRANSACValidation.maximum = 500000000
    maxRANSACValidation.value = 500
    maxRANSACValidation.setToolTip("Maximum number of RANSAC validation steps")
    rigidRegistrationFormLayout.addRow("Maximum RANSAC validation steps: ", maxRANSACValidation)

    # ICP distance threshold slider
    ICPDistanceThreshold = ctk.ctkSliderWidget()
    ICPDistanceThreshold.singleStep = .1
    ICPDistanceThreshold.minimum = 0.1
    ICPDistanceThreshold.maximum = 2
    ICPDistanceThreshold.value = 0.4
    ICPDistanceThreshold.setToolTip("Maximum ICP points-pair distance threshold")
    rigidRegistrationFormLayout.addRow("Maximum ICP distance: ", ICPDistanceThreshold)

    # Alpha slider
    alpha = ctk.ctkDoubleSpinBox()
    alpha.singleStep = .1
    alpha.setDecimals(1)
    alpha.minimum = 0.1
    alpha.maximum = 10
    alpha.value = 2
    alpha.setToolTip("Parameter specifying trade-off between fit and smoothness. Low values induce fluidity, while higher values impose rigidity")
    deformableRegistrationFormLayout.addRow("Rigidity (alpha): ", alpha)

    # Beta slider
    beta = ctk.ctkDoubleSpinBox()
    beta.singleStep = 0.1
    beta.setDecimals(1)
    beta.minimum = 0.1
    beta.maximum = 10
    beta.value = 2
    beta.setToolTip("Width of gaussian filter used when applying smoothness constraint")
    deformableRegistrationFormLayout.addRow("Motion coherence (beta): ", beta)

    # # CPD iterations slider
    CPDIterations = ctk.ctkSliderWidget()
    CPDIterations.singleStep = 1
    CPDIterations.minimum = 100
    CPDIterations.maximum = 1000
    CPDIterations.value = 100
    CPDIterations.setToolTip("Maximum number of iterations of the CPD procedure")
    deformableRegistrationFormLayout.addRow("CPD iterations: ", CPDIterations)

    # # CPD tolerance slider
    CPDTolerence = ctk.ctkSliderWidget()
    CPDTolerence.setDecimals(4)
    CPDTolerence.singleStep = .0001
    CPDTolerence.minimum = 0.0001
    CPDTolerence.maximum = 0.01
    CPDTolerence.value = 0.001
    CPDTolerence.setToolTip("Tolerance used to assess CPD convergence")
    deformableRegistrationFormLayout.addRow("CPD tolerance: ", CPDTolerence)

    return projectionFactor, pointDensity, normalSearchRadius, FPFHSearchRadius, distanceThreshold, maxRANSAC, maxRANSACValidation, ICPDistanceThreshold, alpha, beta, CPDIterations, CPDTolerence
    
#
# MALPACALogic
#

class MALPACALogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
 
  def runLandmarkMultiprocess(self, sourceModelPath, sourceLandmarkPath, targetModelDirectory, outputDirectory, skipScaling, projectionFactor, parameters):
    extensionModel = ".ply"
    specimenOutput = os.path.join(outputDirectory,'individualEstimates')
    medianOutput = os.path.join(outputDirectory,'medianEstimates')
    os.makedirs(specimenOutput, exist_ok=True)
    os.makedirs(medianOutput, exist_ok=True)
    # Iterate through target models
    for targetFileName in os.listdir(targetModelDirectory):
      if targetFileName.endswith(extensionModel):
        targetFilePath = os.path.join(targetModelDirectory, targetFileName)
        #(targetBaseName, targetExt) = os.path.splitext(targetFileName)
        #specimenOutput = os.path.join(outputDirectory, targetBaseName)
        #os.makedirs(specimenOutput, exist_ok=True)
        LM_list = []
        # Iterate through template models
        for file in os.listdir(sourceModelPath):
            if file.endswith(extensionModel):
                sourceFilePath = os.path.join(sourceModelPath,file)
                (baseName, ext) = os.path.splitext(file)
                landmarkFileName = baseName + '.fcsv'
                sourceLandmarkFile = os.path.join(sourceLandmarkPath, landmarkFileName)            
                # Subsample source and target models
                sourceData, targetData, sourcePoints, targetPoints, sourceFeatures, targetFeatures, voxelSize, scaling = self.runSubsample(sourceFilePath, 
                    targetFilePath, skipScaling, parameters)
                # Rigid registration of source sampled points and landmarks
                sourceLM_vtk = self.loadAndScaleFiducials(sourceLandmarkFile, scaling)
                ICPTransform = self.estimateTransform(sourcePoints, targetPoints, sourceFeatures, targetFeatures, voxelSize, parameters)
                ICPTransform_vtk = self.convertMatrixToVTK(ICPTransform)
                sourceSLM_vtk = self.convertPointsToVTK(sourcePoints.points)
                alignedSourceSLM_vtk = self.applyTransform(ICPTransform_vtk, sourceSLM_vtk)
                alignedSourceLM_vtk = self.applyTransform(ICPTransform_vtk, sourceLM_vtk)
            
                # Non-rigid Registration
                alignedSourceSLM_np = vtk_np.vtk_to_numpy(alignedSourceSLM_vtk.GetPoints().GetData())
                alignedSourceLM_np = vtk_np.vtk_to_numpy(alignedSourceLM_vtk.GetPoints().GetData())
                registeredSourceLM_np = self.runCPDRegistration(alignedSourceLM_np, alignedSourceSLM_np, targetPoints.points, parameters)
                outputFiducialNode = self.exportPointCloud(registeredSourceLM_np, "Initial Predicted Landmarks")
                self.RAS2LPSTransform(outputFiducialNode)
                # Projection
                if projectionFactor == 0:
                # Save output landmarks
                    rootName = os.path.splitext(targetFileName)[0]         
                    outputFilePath = os.path.join(specimenOutput, rootName + "_" + baseName + ".fcsv")
                    slicer.util.saveNode(outputFiducialNode, outputFilePath)
                    slicer.mrmlScene.RemoveNode(outputFiducialNode)
                    LM_list.append(registeredSourceLM_np)
                else: 
                    outputPoints_vtk = self.getFiducialPoints(outputFiducialNode)
                    targetModelNode = slicer.util.loadModel(targetFilePath)
                    sourceModelNode = slicer.util.loadModel(sourceFilePath)
                    sourcePoints = slicer.util.arrayFromModelPoints(sourceModelNode)
                    sourcePoints[:] = np.asarray(sourceData.points)
                    sourceModelNode.GetPolyData().GetPoints().GetData().Modified()
                    sourceModelNode_warped = self.applyTPSTransform(sourceLM_vtk.GetPoints(), outputPoints_vtk, sourceModelNode, 'Warped Source Mesh')
                    
                    # project landmarks from template to model
                    maxProjection = (targetModelNode.GetPolyData().GetLength()) * projectionFactor
                    projectedPoints = self.projectPointsPolydata(sourceModelNode_warped.GetPolyData(), targetModelNode.GetPolyData(), outputPoints_vtk, maxProjection)
                    projectedLMNode= slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode',"Refined Predicted Landmarks")
                    for i in range(projectedPoints.GetNumberOfPoints()):
                        point = projectedPoints.GetPoint(i)
                        projectedLMNode.AddFiducialFromArray(point)
                        
                    # Save output landmarks
                    rootName = os.path.splitext(targetFileName)[0]
                    outputFilePath = os.path.join(specimenOutput, rootName + "_" + baseName + ".fcsv")
                    slicer.util.saveNode(projectedLMNode, outputFilePath)
                    slicer.mrmlScene.RemoveNode(outputFiducialNode)
                    slicer.mrmlScene.RemoveNode(projectedLMNode)
                    slicer.mrmlScene.RemoveNode(sourceModelNode)
                    slicer.mrmlScene.RemoveNode(targetModelNode)
                    slicer.mrmlScene.RemoveNode(sourceModelNode_warped)
                    np_array = vtk_np.vtk_to_numpy(projectedPoints.GetPoints().GetData())
                    LM_list.append(np_array)

        median_pos = np.mean(LM_list, axis=0) #np.median(LM_list, axis=0)
        outputMedianNode = self.exportPointCloud(median_pos, "Median Predicted Landmarks")
        rootName = os.path.splitext(targetFileName)[0]         
        outputMedianPath = os.path.join(medianOutput, rootName + "_median.fcsv")
        slicer.util.saveNode(outputMedianNode, outputMedianPath)


          

  def exportPointCloud(self, pointCloud, nodeName):
    fiducialNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode',nodeName)
    for point in pointCloud:
      fiducialNode.AddFiducialFromArray(point) 
    return fiducialNode

    #node.AddFiducialFromArray(point)
  def applyTPSTransform(self, sourcePoints, targetPoints, modelNode, nodeName):
    transform=vtk.vtkThinPlateSplineTransform()  
    transform.SetSourceLandmarks( sourcePoints)
    transform.SetTargetLandmarks( targetPoints )
    transform.SetBasisToR() # for 3D transform
    
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(modelNode.GetPolyData())
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    
    warpedPolyData = transformFilter.GetOutput()
    warpedModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', nodeName)
    warpedModelNode.CreateDefaultDisplayNodes()
    warpedModelNode.SetAndObservePolyData(warpedPolyData)
    #self.RAS2LPSTransform(warpedModelNode)
    return warpedModelNode
      
  def runCPDRegistration(self, sourceLM, sourceSLM, targetSLM, parameters):
    from open3d import geometry
    from open3d import utility
    sourceArrayCombined = np.append(sourceSLM, sourceLM, axis=0)
    targetArray = np.asarray(targetSLM)
    #Convert to pointcloud for scaling
    sourceCloud = geometry.PointCloud()
    sourceCloud.points = utility.Vector3dVector(sourceArrayCombined)
    targetCloud = geometry.PointCloud()
    targetCloud.points = utility.Vector3dVector(targetArray)
    cloudSize = np.max(targetCloud.get_max_bound() - targetCloud.get_min_bound())
    targetCloud.scale(25 / cloudSize, center = (0,0,0))
    sourceCloud.scale(25 / cloudSize, center = (0,0,0))
    #Convert back to numpy for cpd
    sourceArrayCombined = np.asarray(sourceCloud.points,dtype=np.float32)
    targetArray = np.asarray(targetCloud.points,dtype=np.float32)
    registrationOutput = self.cpd_registration(targetArray, sourceArrayCombined, parameters["CPDIterations"], parameters["CPDTolerence"], parameters["alpha"], parameters["beta"])
    deformed_array, _ = registrationOutput.register()
    #Capture output landmarks from source pointcloud
    fiducial_prediction = deformed_array[-len(sourceLM):]
    fiducialCloud = geometry.PointCloud()
    fiducialCloud.points = utility.Vector3dVector(fiducial_prediction)
    fiducialCloud.scale(cloudSize/25, center = (0,0,0))
    return np.asarray(fiducialCloud.points)
    
  def RAS2LPSTransform(self, modelNode):
    matrix=vtk.vtkMatrix4x4()
    matrix.Identity()
    matrix.SetElement(0,0,-1)
    matrix.SetElement(1,1,-1)
    transform=vtk.vtkTransform()
    transform.SetMatrix(matrix)
    transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode', 'RAS2LPS')
    transformNode.SetAndObserveTransformToParent( transform )
    modelNode.SetAndObserveTransformNodeID(transformNode.GetID())
    slicer.vtkSlicerTransformLogic().hardenTransform(modelNode)
    slicer.mrmlScene.RemoveNode(transformNode)
       
  def convertMatrixToVTK(self, matrix):
    matrix_vtk = vtk.vtkMatrix4x4()
    for i in range(4):
      for j in range(4):
        matrix_vtk.SetElement(i,j,matrix[i][j])
    return matrix_vtk
         
  def convertMatrixToTransformNode(self, matrix, transformName):
    matrix_vtk = vtk.vtkMatrix4x4()
    for i in range(4):
      for j in range(4):
        matrix_vtk.SetElement(i,j,matrix[i][j])

    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix_vtk)
    transformNode =  slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode', transformName)
    transformNode.SetAndObserveTransformToParent( transform )
    
    return transformNode
    
  def applyTransform(self, matrix, polydata):
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(polydata)
    transformFilter.Update()
    return transformFilter.GetOutput()
  
  def convertPointsToVTK(self, points): 
    array_vtk = vtk_np.numpy_to_vtk(points, deep=True, array_type=vtk.VTK_FLOAT)
    points_vtk = vtk.vtkPoints()
    points_vtk.SetData(array_vtk)
    polydata_vtk = vtk.vtkPolyData()
    polydata_vtk.SetPoints(points_vtk)
    return polydata_vtk
      
  def displayPointCloud(self, polydata, pointRadius, nodeName, nodeColor):
    #set up glyph for visualizing point cloud
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetRadius(pointRadius)
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(sphereSource.GetOutputPort())
    glyph.SetInputData(polydata)
    glyph.ScalingOff()
    glyph.Update() 
    
    #display
    modelNode=slicer.mrmlScene.GetFirstNodeByName(nodeName)
    if modelNode is None:  # if there is no node with this name, create with display node
      modelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', nodeName)
      modelNode.CreateDefaultDisplayNodes()
    
    modelNode.SetAndObservePolyData(glyph.GetOutput())
    modelNode.GetDisplayNode().SetColor(nodeColor) 
    return modelNode
    
  def displayMesh(self, polydata, nodeName, nodeColor):
    modelNode=slicer.mrmlScene.GetFirstNodeByName(nodeName)
    if modelNode is None:  # if there is no node with this name, create with display node
      modelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', nodeName)
      modelNode.CreateDefaultDisplayNodes()
    
    modelNode.SetAndObservePolyData(polydata)
    modelNode.GetDisplayNode().SetColor(nodeColor) 
    return modelNode
    
  def estimateTransform(self, sourcePoints, targetPoints, sourceFeatures, targetFeatures, voxelSize, parameters):
    ransac = self.execute_global_registration(sourcePoints, targetPoints, sourceFeatures, targetFeatures, voxelSize * 2.5, 
      parameters["distanceThreshold"], parameters["maxRANSAC"], parameters["maxRANSACValidation"])
    
    # Refine the initial registration using an Iterative Closest Point (ICP) registration
    icp = self.refine_registration(sourcePoints, targetPoints, sourceFeatures, targetFeatures, voxelSize * 2.5, ransac, parameters["ICPDistanceThreshold"]) 
    return icp.transformation                                     
  
  def runSubsample(self, sourcePath, targetPath, skipScaling, parameters):
    from open3d import io
    print(":: Loading point clouds and downsampling")
    source = io.read_point_cloud(sourcePath)
    sourceSize = np.linalg.norm(np.asarray(source.get_max_bound()) - np.asarray(source.get_min_bound()))
    target = io.read_point_cloud(targetPath)
    targetSize = np.linalg.norm(np.asarray(target.get_max_bound()) - np.asarray(target.get_min_bound()))
    voxel_size = targetSize/(55*parameters["pointDensity"])
    scaling = (targetSize)/sourceSize
    if skipScaling != 0:
        scaling = 1
    source.scale(scaling, center=(0,0,0))    
    source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size, parameters["normalSearchRadius"], parameters["FPFHSearchRadius"])
    target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size, parameters["normalSearchRadius"], parameters["FPFHSearchRadius"])
    return source, target, source_down, target_down, source_fpfh, target_fpfh, voxel_size, scaling
  
  def loadAndScaleFiducials (self, fiducialPath, scaling): 
    from open3d import geometry
    from open3d import utility
    sourceLandmarkNode =  slicer.util.loadMarkups(fiducialPath)
    self.RAS2LPSTransform(sourceLandmarkNode)
    point = [0,0,0]
    sourceLandmarks_np=np.zeros(shape=(sourceLandmarkNode.GetNumberOfFiducials(),3))
    for i in range(sourceLandmarkNode.GetNumberOfFiducials()):
      sourceLandmarkNode.GetMarkupPoint(0,i,point)
      sourceLandmarks_np[i,:]=point
    slicer.mrmlScene.RemoveNode(sourceLandmarkNode)
    cloud = geometry.PointCloud()
    cloud.points = utility.Vector3dVector(sourceLandmarks_np)
    cloud.scale(scaling, center=(0,0,0))
    fiducialVTK = self.convertPointsToVTK (cloud.points)
    return fiducialVTK

  def distanceMatrix(self, a):
    """
    Computes the euclidean distance matrix for n points in a 3D space
    Returns a nXn matrix
     """
    id,jd=a.shape
    fnx = lambda q : q - np.reshape(q, (id, 1))
    dx=fnx(a[:,0])
    dy=fnx(a[:,1])
    dz=fnx(a[:,2])
    return (dx**2.0+dy**2.0+dz**2.0)**0.5
    
  def preprocess_point_cloud(self, pcd, voxel_size, radius_normal_factor, radius_feature_factor):
    from open3d import geometry
    from open3d import registration
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * radius_normal_factor
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * radius_feature_factor
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = registration.compute_fpfh_feature(
        pcd_down,
        geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh



  def execute_global_registration(self, source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size, distance_threshold_factor, maxIter, maxValidation):
    from open3d import registration
    distance_threshold = voxel_size * distance_threshold_factor
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        registration.TransformationEstimationPointToPoint(True), 4, [
            registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], registration.RANSACConvergenceCriteria(maxIter, maxValidation))
    return result


  def refine_registration(self, source, target, source_fpfh, target_fpfh, voxel_size, result_ransac, ICPThreshold_factor):
    from open3d import registration
    distance_threshold = voxel_size * ICPThreshold_factor
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        registration.TransformationEstimationPointToPlane())
    return result
    
  def cpd_registration(self, targetArray, sourceArray, CPDIterations, CPDTolerence, alpha_parameter, beta_parameter):
    from pycpd import DeformableRegistration
    output = DeformableRegistration(**{'X': targetArray, 'Y': sourceArray,'max_iterations': CPDIterations, 'tolerance': CPDTolerence, 'low_rank':True}, alpha = alpha_parameter, beta  = beta_parameter)
    return output
    
  def getFiducialPoints(self,fiducialNode):
    points = vtk.vtkPoints()
    point=[0,0,0]
    for i in range(fiducialNode.GetNumberOfFiducials()):
      fiducialNode.GetNthFiducialPosition(i,point)
      points.InsertNextPoint(point)
    
    return points
    
  def runPointProjection(self, template, model, templateLandmarks, maxProjectionFactor):
    maxProjection = (model.GetPolyData().GetLength()) * maxProjectionFactor
    print("Max projection: ", maxProjection)
    templatePoints = self.getFiducialPoints(templateLandmarks)
      
    # project landmarks from template to model
    projectedPoints = self.projectPointsPolydata(template.GetPolyData(), model.GetPolyData(), templatePoints, maxProjection)
    projectedLMNode= slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode',"Refined Predicted Landmarks")
    for i in range(projectedPoints.GetNumberOfPoints()):
      point = projectedPoints.GetPoint(i)
      projectedLMNode.AddFiducialFromArray(point)
    return projectedLMNode
  
  def projectPointsPolydata(self, sourcePolydata, targetPolydata, originalPoints, rayLength):
    print("original points: ", originalPoints.GetNumberOfPoints())
    #set up polydata for projected points to return
    projectedPointData = vtk.vtkPolyData()
    projectedPoints = vtk.vtkPoints()
    projectedPointData.SetPoints(projectedPoints)
    
    #set up locater for intersection with normal vector rays
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(targetPolydata)
    obbTree.BuildLocator()
    
    #set up point locator for finding surface normals and closest point
    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(sourcePolydata)
    pointLocator.BuildLocator()
    
    targetPointLocator = vtk.vtkPointLocator()
    targetPointLocator.SetDataSet(targetPolydata)
    targetPointLocator.BuildLocator()
    
    #get surface normal from each landmark point
    rayDirection=[0,0,0]
    normalArray = sourcePolydata.GetPointData().GetArray("Normals")
    if(not normalArray):
      print("no normal array, calculating....")
      normalFilter=vtk.vtkPolyDataNormals()
      normalFilter.ComputePointNormalsOn()
      normalFilter.SetInputData(sourcePolydata)
      normalFilter.Update()
      normalArray = normalFilter.GetOutput().GetPointData().GetArray("Normals")
      if(not normalArray):
        print("Error: no normal array")
        return projectedPointData
    for index in range(originalPoints.GetNumberOfPoints()):
      originalPoint= originalPoints.GetPoint(index)
      # get ray direction from closest normal
      closestPointId = pointLocator.FindClosestPoint(originalPoint)
      rayDirection = normalArray.GetTuple(closestPointId)
      rayEndPoint=[0,0,0]
      for dim in range(len(rayEndPoint)):
        rayEndPoint[dim] = originalPoint[dim] + rayDirection[dim]* rayLength
      intersectionIds=vtk.vtkIdList()
      intersectionPoints=vtk.vtkPoints()
      obbTree.IntersectWithLine(originalPoint,rayEndPoint,intersectionPoints,intersectionIds)
      #if there are intersections, update the point to most external one.
      if intersectionPoints.GetNumberOfPoints() > 0:
        exteriorPoint = intersectionPoints.GetPoint(intersectionPoints.GetNumberOfPoints()-1)
        projectedPoints.InsertNextPoint(exteriorPoint)
      #if there are no intersections, reverse the normal vector
      else: 
        for dim in range(len(rayEndPoint)):
          rayEndPoint[dim] = originalPoint[dim] + rayDirection[dim]* -rayLength
        obbTree.IntersectWithLine(originalPoint,rayEndPoint,intersectionPoints,intersectionIds)
        if intersectionPoints.GetNumberOfPoints()>0:
          exteriorPoint = intersectionPoints.GetPoint(0)
          projectedPoints.InsertNextPoint(exteriorPoint)
        #if none in reverse direction, use closest mesh point
        else:
          closestPointId = targetPointLocator.FindClosestPoint(originalPoint)
          rayOrigin = targetPolydata.GetPoint(closestPointId)
          projectedPoints.InsertNextPoint(rayOrigin)
    return projectedPointData
    
  def takeScreenshot(self,name,description,type=-1):
    # show the message even if not taking a screen shot
    slicer.util.delayDisplay('Take screenshot: '+description+'.\nResult is available in the Annotations module.', 3000)
    
    lm = slicer.app.layoutManager()
    # switch on the type to get the requested window
    widget = 0
    if type == slicer.qMRMLScreenShotDialog.FullLayout:
      # full layout
      widget = lm.viewport()
    elif type == slicer.qMRMLScreenShotDialog.ThreeD:
      # just the 3D window
      widget = lm.threeDWidget(0).threeDView()
    elif type == slicer.qMRMLScreenShotDialog.Red:
      # red slice window
      widget = lm.sliceWidget("Red")
    elif type == slicer.qMRMLScreenShotDialog.Yellow:
      # yellow slice window
      widget = lm.sliceWidget("Yellow")
    elif type == slicer.qMRMLScreenShotDialog.Green:
      # green slice window
      widget = lm.sliceWidget("Green")
    else:
      # default to using the full window
      widget = slicer.util.mainWindow()
      # reset the type so that the node is set correctly
      type = slicer.qMRMLScreenShotDialog.FullLayout
    
    # grab and convert to vtk image data
    qimage = ctk.ctkWidgetsUtils.grabWidget(widget)
    imageData = vtk.vtkImageData()
    slicer.qMRMLUtils().qImageToVtkImageData(qimage,imageData)
    
    annotationLogic = slicer.modules.annotations.logic()
    annotationLogic.CreateSnapShot(name, description, type, 1, imageData)


class MALPACATest(ScriptedLoadableModuleTest):
  """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
  
  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
      """
    slicer.mrmlScene.Clear(0)
  
  def runTest(self):
    """Run as few or as many tests as needed here.
      """
    self.setUp()
    self.test_MALPACA1()
  
  def test_MALPACA1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
      tests should exercise the functionality of the logic with different inputs
      (both valid and invalid).  At higher levels your tests should emulate the
      way the user would interact with your code and confirm that it still works
      the way you intended.
      One of the most important features of the tests is that it should alert other
      developers when their changes will have an impact on the behavior of your
      module.  For example, if a developer removes a feature that you depend on,
      your test should break so they know that the feature is needed.
      """
    
    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import urllib
    downloads = (
                 ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
                 )
    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        logging.info('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        logging.info('Loading %s...' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading')
    
    volumeNode = slicer.util.getNode(pattern="FA")
    logic = MALPACALogic()
    self.assertIsNotNone( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')



