
fn cleanDateString = (
	t = localTime
	t = substituteString t "/" "_"
	t = substituteString t ":" "_"
	t = substituteString t " " "_"
	t
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

fn deserializeMatrix3 str = (
	local p = filterString str ","
	local row1 = [strToFloat p[1], strToFloat p[2], strToFloat p[3]]
	local row2 = [strToFloat p[4], strToFloat p[5], strToFloat p[6]]
	local row3 = [strToFloat p[7], strToFloat p[8], strToFloat p[9]]
	local row4 = [strToFloat p[10], strToFloat p[11], strToFloat p[12]]
	return matrix3 row1 row2 row3 row4
)

fn deleteMissingCustAttributes = (
	cas = getclassinstances Missing_Custom_Attribute_Plugin
	deleted = [cas.count, 0]
	for ca in cas do (
		owner = custattributes.getowner ca
		if isvalidobj owner do (
			n = finditem owner.custattributes ca
			if (n > 0) do (
				deleteitem owner.custattributes n
				deleted[2] += 1
			)
		)
	)
	deleted
)

fn findParentFolderOfScript = (
	parentFolder = getThisScriptFilename() 
	local i = 0
	while i < 3 do (
		parentFolder = pathConfig.removePathLeaf parentFolder
		i += 1
	)
	print ("parentFolder of script: " + parentFolder as string)
	parentFolder as string
)

fn findScriptFolder = (
	scriptFolder = getThisScriptFilename()
	scriptFolder = pathConfig.removePathLeaf scriptFolder
	print ("scriptFolder of script: " + scriptFolder as string)
	scriptFolder
)

fn getToExportDataFilepath = (
	dataFilepathsFilepath = findScriptFolder() + "\\dataFilepaths.txt"

	local dataFilepathsFile = openFile dataFilepathsFilepath mode:"r"
	linesArray = #()
	if dataFilepathsFile != undefined then (
		while not eof dataFilepathsFile do (
			line1 = readLine dataFilepathsFile
			append linesArray line1
			print ("line " + line1)
		)
		close dataFilepathsFile

		local nodesToDetachFilepath = substitutestring linesArray[1] ".txt" "_NodesToDetach.txt"
		append linesArray nodesToDetachFilepath
	) else (
		print "dataFilepathsFile is undefined"
	)
	linesArray
)

fn detachNodes toDetachFilepath = (
	local toDetachFile = openFile toDetachFilepath mode:"r"
	if toDetachFile != undefined then (
		while not eof toDetachFile do (
			
			-- get the name of the node to detach
			toDetachLine = readLine toDetachFile
			nodeToDetach = getNodeByName toDetachLine

			-- get the head of its group so you can "open" it
			groupHead = nodeToDetach
			while groupHead.parent != undefined do groupHead = groupHead.parent
			groupHeadName = groupHead.name as string
			
			-- open the group with the group head
			clearSelection()
			select groupHead
			for c in groupHead.children do select c
			max group open

			-- select and detach the node
			clearSelection()
			select nodeToDetach
			max group detach
			clearSelection()

			-- close group again
			groupHeadAttemptedReselect = getNodeByName groupHeadName
			if groupHeadAttemptedReselect != undefined do (
				select groupHeadAttemptedReselect
				for c in groupHeadAttemptedReselect.children do select c
				max group close
				clearSelection()
			)	
		)
		close toDetachFile
	) else (
		throw "toDetachFile undefined see fn detachNodes"
	)
)

fn writeToDataFilepath rootDirectory = (
	dataFilepathsFilepath = findScriptFolder() + "\\dataFilepaths.txt"

	local dataFilepathsFile = openFile dataFilepathsFilepath mode:"w"
	if dataFilepathsFile != undefined then (
		format "%" rootDirectory to:dataFilepathsFile
	) else (
		print "dataFilepathsFile is undefined"
	)
	close dataFilepathsFile
)

fn createSceneDirectory sceneName = (
	directory = findParentFolderOfScript() + "\\output\\instancesExported\\01_29_2019_testing_set\\"	
	os = python.import("os")
	t = cleanDateString()
	directory = directory + t + "-" + sceneName + "\\" 
	if os.path.exists(directory) == false do (
		os.mkdir(directory)
		print ("made directory " + directory as string)
	)
	directory
)

fn makeMaxFileName dir itemName = (
	nameOfMaxFile = "script_created_" + itemName + ".max"
	directory = dir + "\\"
	filepathWithName = directory + nameOfMaxFile
	filepathWithName
)

(
		--get the toExportDataFilepath from the dataFilepathsFile
		linesArray = getToExportDataFilepath()
		if linesArray.count > 2 then (
			global sceneName = linesArray[2]
			local toExportDataFilepath = linesArray[1]
			detachNodes linesArray[3]
		) else (
			throw "error reading dataFilepaths text file in script #2 export instances"
		)

		--set up directories and filepaths
		directory = createSceneDirectory sceneName
		uuidGenerator = python.import("uuid")
		rootDirectory = directory 
		uuidFilepath = rootDirectory + "\\uuidFile.txt"
		local uuidFile = createfile uuidFilepath
		
		--export data into folders
		local file1 = openFile toExportDataFilepath
		while not eof file1 do (
			id = uuidGenerator.uuid4()
			directory = rootDirectory + "\\" + id + "\\"
			os = python.import("os")
			if os.path.exists(directory) == false do (
				os.mkdir(directory)
			)
			
			r = readLine file1
			print r
			
			-- if the line has matches (it has a comma)
			if findString r "," != undefined then (
				--parse line with match,submatch,submatch, etc...
				matches = filterString r ","
			
				if matches.count > 0 do (
					nodeToExport = getNodeByName matches[1] -- maxfile to export

					select nodeToExport
					if selection.count > 0 do (
						
						-- make filepath
						nameAsString = nodeToExport.name as string
						nameAsString = substituteString nameAsString " " ""
						filepathWithName = makeMaxFileName directory nameAsString
						
						-- write match to uuid file
						originalName = nodeToExport.name
						originalNameString = originalName as string
						format "%:%\n" originalNameString id to:uuidFile
						nodeToExport.name = "main_node"
		
						-- save the node to the directory
						for c in nodeToExport.children do select c
						saveNodes selection filepathWithName quiet:true
						print ("exported " + originalNameString + " to " + directory)
						
						-- restore the original name
						print ("originalName to restore: " + originalNameString)
						nodeToExport.name = originalName
						
						-- save all the transform data
						filepathWithName = directory + nameAsString + "_match_transform_data.txt"
						local dataFile = createfile filepathWithName
						for objName in matches do (
							obj = getNodeByName objName
							transformationMatrix = obj.transform
							transformationMatrixAsString = serializeMatrix3 transformationMatrix
							format "%:%\n" objName transformationMatrixAsString to:dataFile
						)
						close dataFile
					)	
				)
			) else (
				-- there were no matches so just export?
				nodeToExport = getNodeByName r -- maxfile to export
				select nodeToExport

				if nodeToExport == undefined or selection == undefined do (
					print ("nodeToExport or selection is undefined " + r)
					continue
				) 

				if selection != undefined and selection.count > 0 do (
					
					-- make filepath
					nameAsString = nodeToExport.name as string
					nameAsString = substituteString nameAsString " " "_"
					filepathWithName = makeMaxFileName directory nameAsString
					print (nameAsString + " will be to max file: " + filepathWithName)
					
					-- write match to uuid file
					originalName = nameAsString
					originalNameString = originalName as string
					originalNameString = substituteString originalNameString " " "_"
					format "%:%\n" originalNameString id to:uuidFile
					nodeToExport.name = "main_node"
					print ("named main_node")
					
					-- save the node to the directory
					for c in nodeToExport.children do select c
					print "selected children"
					try (
						saveNodes selection filepathWithName quiet:true

						print "savedNodes"
						print ("originalName to restore: " + originalNameString)
						nodeToExport.name = originalName	

						-- save transform to transform file
						filepathWithName = directory + originalNameString + "_match_transform_data.txt"
						local dataFile = createfile filepathWithName
						transformationMatrix = nodeToExport.transform
						transformationMatrixAsString = serializeMatrix3 transformationMatrix
						format "%:%\n" nodeToExport.name transformationMatrixAsString to:dataFile
						close dataFile
					) catch (
						print ("unable to save " + originalNameString)
					)
				)
			)
		)
		close file1
		close uuidFile
		writeToDataFilepath rootDirectory
		print "done with script #2 export instances\n"		
)		