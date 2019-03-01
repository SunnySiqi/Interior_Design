fn getBBoxPoints obj = (
	local bbPoints = #()

	local bbMax = obj.max
	local bbMin = obj.min

	append bbPoints (Point3 bbMax.x bbMax.y bbMax.z)	-- top-right-front corner
	append bbPoints (Point3 bbMax.x bbMin.y bbMax.z)	-- top-right-back corner
	append bbPoints (Point3 bbMin.x bbMin.y bbMax.z)	-- top-left-back corner
	append bbPoints (Point3 bbMin.x bbMax.y bbMax.z)	-- top-left-front corner
	
	append bbPoints (Point3 bbMax.x bbMax.y bbMin.z)	-- bottom-right-front corner
	append bbPoints (Point3 bbMax.x bbMin.y bbMin.z)	-- bottom-right-back corner
	append bbPoints (Point3 bbMin.x bbMin.y bbMin.z)	    -- bottom-left-back corner
	append bbPoints (Point3 bbMin.x bbMax.y bbMin.z)	-- bottom-left-front corner
	
	return bbPoints
)


fn getSize boundingBoxPoints = (
	local size = #()

	if boundingBoxPoints.count == 8 do (
		local depth = abs (boundingBoxPoints[1].y - boundingBoxPoints[2].y)
		local width = abs (boundingBoxPoints[4].x - boundingBoxPoints[1].x)
		local height = abs (boundingBoxPoints[1].z - boundingBoxPoints[5].z)

		if depth == 0.0 do depth = 1.0
		if width == 0.0 do width = 1.0
		if height == 0.0 do height = 1.0
		append size (depth)
		append size (width)
		append size (height)
	)

	return size
)


(
	PreObjects = objects as array
	allObjects = #()

	for obj in PreObjects do (
		objNameString = obj.name as string
		if (findString objNameString "LARK" != undefined) do (

			print "lark in pre objects"
			objParentNameString = obj.parent as string
			print objParentNameString
		)
		if obj.parent == undefined do append allObjects obj
	)

	objCheck1 = #()
	objCheck2 = #()

	depthList = #()
	widthList = #()
	heightList = #()
	arch = #()
	
	sumD = 0.0
	sumW = 0.0
	sumH = 0.0
	countD = 0

	

	for obj in allObjects do (
		boundingBoxPoints = getBBoxPoints obj
		size = getSize boundingBoxPoints
		found = 0

		objNameString = obj.name as string
		if (findString objNameString "LARK" != undefined) do (
			print "lark in all objects"
			found = 1
		)

		if size[1] < 0.1 or size[1] >1000 do (
			print obj.name
			append arch obj

			if found == 1 do print "first continue"

			continue
		)
		
		if size[2] < 0.1 or size[2] >1000 do (
			print obj.name
			append arch obj
			if found == 1 do print "second continue"
			continue
		)
		
		if size[3] < 0.1 or size[3] >1000 do (
			print obj.name
			append arch obj
			if found == 1 do print "third continue"
			continue
		)
		if found == 1 do print "no continues"

		append depthList size[1]
		sumD = sumD + size[1]
		append widthList size[2]
		sumW = sumW + size[2]
		append heightList size[2]
		sumH = sumH + size[2]
		append objCheck1 obj
	)
	
	avgD = sumD/depthList.count
	print ("avgD:" + avgD as string)
	avgW = sumW/widthList.count
	print ("avgW:" + avgW as string)
	avgH = sumH/heightList.count
	print ("avgH:" + avgH as string)
	
	for obj in objCheck1 do(
		boundingBoxPoints = getBBoxPoints obj
		size = getSize boundingBoxPoints
		found = 0

		objNameString = obj.name as string
		if (findString objNameString "Group3993" != undefined) do (
			print "Group3993 in objCheck1"
			found = 1
		)

		if size[1] > 8*avgD or size[2] > 8*avgW or size[3] > 8*avgH then(
			print obj.name as string
			append arch obj
			if found == 1 do print "append Group3993 to arch and continue dont add chair to objCheck2"
			continue
		)
		append objCheck2 obj
		if found == 1 do print "append Group3993 to objCheck2"



		
		
	)
	clearSelection()
	select objCheck2
	arrayToExport = selection as array
	saveNodes arrayToExport "C:\\Users\\Siqi Wang\\Desktop\\scene\\testOutput\\furniture.max" quiet:true

	clearSelection()
	select arch
	arrayToExport = selection as array
	saveNodes arrayToExport "C:\\Users\\Siqi Wang\\Desktop\\scene\\testOutput\\arch.max" quiet:true

/*

	
		--meshObj = obj.mesh
		--if obj.numverts != undefined do print (obj.name as string + " " + meshObj.numverts as string)
*/
	print "done"
)