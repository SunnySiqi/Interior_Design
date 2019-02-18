-- directory navigation
-- 1. provide C:\Users\Montana Fowler\Desktop\scripts\output\instances\testing9\11_28_2018_4_29_43_PM
-- 		it is written into DataFilepaths automatically if you run scripts 0-2 first, otherwise you'll need to manually set the directory
-- 2. navigate folders, each is a unique id for the unique instance of the object
-- 		3. when we are in a folder, get the max node data from the max file... (?) "mergeMaxFile" ? test this
-- 		4. read the transform data to clone those nodes and apply that transform to them - easy

fn strToFloat str = (
	return (str as float)
)

fn selectNodeAndChildren nodeToSelect = (
	select nodeToSelect
	--print ("selectNodeAndChildren " + nodeToSelect.Name as string)
	for child in nodeToSelect.Children do selectNodeAndChildren child
)

fn deserializeMatrix3 str = (
	local p = filterString str ","
	local row1 = [strToFloat p[1], strToFloat p[2], strToFloat p[3]]
	local row2 = [strToFloat p[4], strToFloat p[5], strToFloat p[6]]
	local row3 = [strToFloat p[7], strToFloat p[8], strToFloat p[9]]
	local row4 = [strToFloat p[10], strToFloat p[11], strToFloat p[12]]
	return matrix3 row1 row2 row3 row4
)

fn serializeMatrix3 mat3 = (
	local ss = stringstream ""
	format "%,%,%,%,%,%,%,%,%,%,%,%" \
		(mat3.row1.x) \
		(mat3.row1.y) \
		(mat3.row1.z) \
		(mat3.row2.x) \
		(mat3.row2.y) \
		(mat3.row2.z) \
		(mat3.row3.x) \
		(mat3.row3.y) \
		(mat3.row3.z) \
		(mat3.row4.x) \
		(mat3.row4.y) \
		(mat3.row4.z) \
		to:ss
	return ss as string
)

fn deleteMissingCustAttributes = 
(
	cas = getclassinstances Missing_Custom_Attribute_Plugin
	deleted = [cas.count, 0]
	for ca in cas do
	(
		owner = custattributes.getowner ca
		if isvalidobj owner do
		(
			n = finditem owner.custattributes ca
			if (n > 0) do 
			(
				deleteitem owner.custattributes n
				deleted[2] += 1
			)
		)
	)
	deleted
)

fn findScriptFolder = (
	scriptFolder = getThisScriptFilename()
	scriptFolder = pathConfig.removePathLeaf scriptFolder
	scriptFolder
)

fn getSceneInstancesDirectory = (
	dataFilepathsFilepath = findScriptFolder() + "\\dataFilepaths.txt"

	local dataFilepathsFile = openFile dataFilepathsFilepath mode:"r"
	linesArray = #()
	if dataFilepathsFile != undefined then (
		while not eof dataFilepathsFile do (
			line1 = readLine dataFilepathsFile
			append linesArray line1
		)
		close dataFilepathsFile
	) else (
		throw "dataFilepathsFile is undefined"
	)
	substituteString linesArray[1] "\n" ""
)

(
	-- compile directories of maxfiles to import
	os = python.import("os")

	-- gets the directory filepath from script #2 writing it to dataFilepaths.txt
	directory = getSceneInstancesDirectory()
		-- you can manually set the directory here if not from running scripts 0-2
		--directory = "C:\Users\Montana Fowler\Desktop\interior_design_project\output\instancesExported\01_29_2019_testing_set\2_7_2019_1_54_18_PM-2866__DR_EVENTS_RENDER_US_113500Holiday-cleaning\"
	
	directories = GetDirectories (directory + "*")
	match = undefined
	for d in directories do (
		join directories (getDirectories (d + "\\*"))
	)
	append directories (directory + "\\") 	
	local oldNodes

	-- for each directory
	for folder in directories do (
		maxfiles = getFiles(folder + "*.max")
		folderParsed = filterString folder "\\"

		for file in maxfiles do (
			-- get all the nodes currently in the scene
			oldNodes = objects as array
			
			-- Try to merge the file, if it can't use the importfile interface
			if (mergeMAXFile file #noRedraw #select #mergeDups #useSceneMtlDups #AutoRenameDups #neverReparent quiet:true) == false then (
				importFile file #noPrompt
			)

			-- collect the new nodes just merged in (they are not in oldNodes array)
			mergedNodes = for obj in objects where findItem oldNodes obj == 0 collect obj
			
			-- get the transform data files for this folder
			transformDataFiles = getFiles(folder + "*match_transform_data.txt")
			
			-- get the node just merged in, called main_node by default
			nodeToDuplicate = getNodeByName "main_node"
			clearSelection()
			
			-- if main_node exists in .max file
			if nodeToDuplicate != undefined do (
				select nodeToDuplicate

				for transformDataFile in transformDataFiles do (
					local transformDataFile1 = openFile transformDataFile
					transformFileLines = #()
					while not eof transformDataFile1 do (
						str = readLine transformDataFile1
						append transformFileLines str
					)
					local index = 0
					for transformFileLine in transformFileLines do (
						
						-- parse the line in the transform  "name:transform matrix"
						itemString = transformFileLine
						parsedItemString = filterString itemString ":"
						index = index + 1
						if parsedItemString.count > 1 and nodeToDuplicate != undefined do (
							-- item's name, make with unique id
							uniqueId = folderParsed[folderParsed.count]
							newName = parsedItemString[1] + "_" + uniqueId

							-- get item's transformation
							transformationMatrix = deserializeMatrix3 parsedItemString[2]
							
							-- select the node to duplicate
							clearSelection()
							select nodeToDuplicate
							
							-- clone & set transform
							if transformFileLines.count > 1 then (
								print ("going to clone " + newName)
								result = #()
								maxOps.cloneNodes nodeToDuplicate offset:[0,0,0] cloneType:#instance actualNodeList:#() newNodes:&result
								the_result = result[1]

								--if findString newName "Plane001_" == undefined and findString newName "Box003_" == undefined do the_result.transform = transformationMatrix
								the_result.transform = transformationMatrix
								the_result.name = newName + "_copy_" + index as string
							) else (
								print ("did not clone " + newName)
								nodeToDuplicate.name = newName
							)
							clearSelection()
						)					
					)
					close transformDataFile1

				)
				-- delete the instance we were cloning, because its transform isnt right
				clearSelection()
				nodeToDuplicate = getNodeByName "main_node"
				if nodeToDuplicate != undefined do (
					selectNodeAndChildren nodeToDuplicate
					select nodeToDuplicate
					for selectedItem in selection do (
						delete selectedItem
					)
				)
			)
		)
	)
	print "done with script #3 reconstruction"
)