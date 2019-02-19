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
	allObjects = objects as array
	
	objCheck1 = #()
	objCheck2 = #()

	depthList = #()
	widthList = #()
	heightList = #()
	sumD = 0.0
	sumW = 0.0
	sumH = 0.0
	countD = 0

	for obj in allObjects do (
		boundingBoxPoints = getBBoxPoints obj
		size = getSize boundingBoxPoints
		if size[1] < 0.1 or size[1] >1000 do (
			print obj.name
			continue
		)
		
		if size[2] < 0.1 or size[2] >1000 do (
			print obj.name
			continue
		)
		
		if size[3] < 0.1 or size[3] >1000 do (
			print obj.name
			continue
		)
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
		if size[1] > 2*avgD or size[2] > 2*avgW or size[3] > 2*avgH then(
			print obj.name as string
			continue
		)
		append objCheck2 obj
		
		
	)


/*

	
		--meshObj = obj.mesh
		--if obj.numverts != undefined do print (obj.name as string + " " + meshObj.numverts as string)
*/
	print "done"
)