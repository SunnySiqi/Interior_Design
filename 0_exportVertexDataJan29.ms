
fn getVertexString obj = (
	
	local vertexKey = ""
	try (
		if obj.numverts > 0 do (
			local vertCount = 3
			if obj.numverts < 3 do (
				vertCount = obj.numverts
			)
			
			for v = 1 to vertCount do (
				local vertex = in coordsys local getvert obj.mesh v
				vertexKey = vertexKey + vertex as string
			)
		)
	) catch (
		--format "%\n" (getCurrentException())
	)	
	vertexKey
)


fn cleanDateString = (
	t = localTime
	t = substituteString t "/" "_"
	t = substituteString t ":" "_"
	t = substituteString t " " "_"
	t
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

(
	allObjects = $*
	local dir = findParentFolderOfScript() + "\\output\\vertexData"
	
	-- make the directory if it does not exist
	/*os = python.import("os")
	if os.path.exists(directory) == false do (
		os.mkdir(directory)
		print ("made directory " + directory as string)
	)*/

	t = cleanDateString()
	sceneName = maxfilename
	sceneName = substituteString sceneName ".max" ""
	sceneName = substituteString sceneName " " ""
	dir = dir + "\\" + t + "-" + sceneName + ".txt" 

	try (
		newFile = createfile dir
		for item in allObjects do (
			vertexString = getVertexString item
			vertexStringFormatted = vertexString as string
			if(vertexStringFormatted.count > 0) do (
				format "%: %\n" item.Name vertexStringFormatted to:newFile
			)
		)
		close newFile
		print ("exported vertex data to " + dir)
	) catch (
		format "*** % ***\n" (getCurrentException())
	)	



	scriptFolder = findScriptFolder()
	dataFilepathsFileFilepath = scriptFolder + "\\dataFilepaths.txt"
	dataFilepathsFile = openFile dataFilepathsFileFilepath mode:"w"
	format "%\n" dir to:dataFilepathsFile
	format "%\n" sceneName to:dataFilepathsFile
	close dataFilepathsFile

	print "done with script #0 export vertex data\n"

	matchingAlgorithmFilepath = scriptFolder + "\\1_treeMatchingAlgorithmJan29.py"
	python.ExecuteFile matchingAlgorithmFilepath
)

