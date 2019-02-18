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
	append bbPoints (Point3 bbMin.x bbMin.y bbMin.z)	-- bottom-left-back corner
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

	local depthList = #()
	local widthList = #()
	local heightList = #()

	for obj in allObjects do (
		print obj.name as string
		print obj.baseObject as string
		boundingBoxPoints = getBBoxPoints obj
		size = getSize boundingBoxPoints
		append depthList size[1]
		append widthList size[2]
		append heightList size[3]
	)

	for depth in depthList do(
		print depth as string
	)

	for width in widthList do(
		print width as string
	)

	for height in heightList do(
		print height as string
	)

	
		--meshObj = obj.mesh
		--if obj.numverts != undefined do print (obj.name as string + " " + meshObj.numverts as string)

	print "done"
)