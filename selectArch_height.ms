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
	PreObjects = objects as array
	allObjects = #()

	for obj in PreObjects do (
		if obj.parent == undefined do(
			obj_type = obj.baseObject as string
			index = findString obj_type "VRay"
			if index == undefined do 
				append allObjects obj	
		) 
	)

	arch = #()
	objCheck = #()
	dMax = 0
	wMax = 0
	hMax = 0
		

	for obj in allObjects do (
		boundingBoxPoints = getBBoxPoints obj
		size = getSize boundingBoxPoints
		if size[1] > dMax and size[1] < 500 do
			dMax = size[1]
		if size[2] > wMax and size[2] < 500 do
			wMax = size[2]
		if size[3] > hMax and size[3] < 500 do
			hMax = size[3]
	)
	print dMax
	print wMax
	print hMax

	for obj in allObjects do (
		boundingBoxPoints = getBBoxPoints obj
		size = getSize boundingBoxPoints
		if size[1] > 0.7* dMax or size[2] > 0.7* wMax or size[3] > 0.7* hMax then(
			print("arch" + obj.name as string)
			append arch obj
		)
		else(
			print("obj" + obj.name as string)
			append objCheck obj	
		)			
	)
	

	clearSelection()
	select arch
	arrayToExport = selection as array
	saveNodes arrayToExport "C:\\Users\\Siqi Wang\\Desktop\\scene\\testOutput\\arch_living_height.max" quiet:true

	
	clearSelection()
	select objCheck
	arrayToExport = selection as array
	saveNodes arrayToExport "C:\\Users\\Siqi Wang\\Desktop\\scene\\testOutput\\objCheck_living_height.max" quiet:true
	
	print "done"
)