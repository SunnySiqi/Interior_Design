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
-- To check if the object is wall: at least 2 dimensions are large and one dimension is small.
fn checkWall objSize = (
	if objSize[1]<35 and objSize[2] >100 and objSize[3] >100 or objSize[2]<35 and objSize[1] >100 and objSize[3] >100 or objSize[3]<35 and objSize[1] >100 and objSize[2] >100 then 
		return true
	else
		return false
)

-- To find architecture related to the wall : Bounding box of the wall contains the bounding box of another object
fn checkWithinWall obj arch = (
	boxArch = box3 arch.max arch.min
	boxObj = box3 obj.max obj.min
	if contains boxArch boxObj then
		return true
	else
		return false	
)

-- If the walls of the room are connected together as a whole: 3 dimensions are large and also in the valid range (not e^30...)
fn checkWholeArch sizeObj = (
	if sizeObj[1] > 100 and sizeObj[2] > 100 and sizeObj[3] > 100 and sizeObj[1] < 500 and sizeObj[2] < 500 and sizeObj[3] < 500 then
		return true
	else
		return false
)

(
	PreObjects = objects as array
	allObjects = #()

	-- Pick the objects of parent node and ignore the camera stuff(with substring "VRay")
	for obj in PreObjects do (
		if obj.parent == undefined do(
			obj_type = obj.baseObject as string
			index = findString obj_type "VRay"
			if index == undefined do 
				append allObjects obj
		) 
	)

	arch = #() --architecture
	objCheck = #() --final objects
	objCheck_half = #() --objects & stuffs related to wall
	flag = 0 -- If the walls are connected as the dinning room scene, then flag is 1, else the wall is separated and flag is 0
		

	-- Find the connected walls in objects -> set flag=1 and define the room size with depthMax, widthMax and HightMax
	for obj in allObjects do(
		boundingBoxPoints = getBBoxPoints obj
		sizeObj = getSize boundingBoxPoints
		if checkWholeArch sizeObj do(
			dMax = size[1]
			wMax = size[2]
			hMax = size[3]
			flag = 1
			break
		)			
	)
	
	/*
	##########################################
	if flag == 1:
		use the depth, width, height of the room to find window frame stuffs. The threshold used is 0.7*Max
	else:
		find the walls
		use the wall and relation "contains" to find window frame stuffs.
	##########################################
	Problems:
	for obj2 in objCheck_half do (
			boundingBoxPoints = getBBoxPoints obj2
			sizeObj2 = getSize boundingBoxPoints
			if sizeObj2[1] > 50 and sizeObj2[1] < 500 or sizeObj2[2] > 50 and sizeObj2[2] < 500 or sizeObj2[3] > 50 and sizeObj2[3] < 500 then(
				for archs in arch do(
					if checkWithinWall obj2 archs then
						append arch obj2
					else
						append objCheck obj2
				)
			)
			else
				append objCheck obj2
		)
	code above runs really slow....(haven't finished for once)
	need more optimization?? now: when the object doesn't have one dimension large enough -> no need to checkWithinWall
	*/
	if flag == 1 then (
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
	)
	else(
		for obj in allObjects do (
			boundingBoxPoints = getBBoxPoints obj
			size = getSize boundingBoxPoints
			if checkWall size then(
				print("arch" + obj.name as string)
				append arch obj
			)
			else
				append objCheck_half obj
		)
		
		for obj2 in objCheck_half do (
			boundingBoxPoints = getBBoxPoints obj2
			sizeObj2 = getSize boundingBoxPoints
			if sizeObj2[1] > 50 and sizeObj2[1] < 500 or sizeObj2[2] > 50 and sizeObj2[2] < 500 or sizeObj2[3] > 50 and sizeObj2[3] < 500 then(
				for archs in arch do(
					if checkWithinWall obj2 archs then
						append arch obj2
					else
						append objCheck obj2
				)
			)
			else
				append objCheck obj2
		)

	)

	clearSelection()
	select arch
	arrayToExport = selection as array
	saveNodes arrayToExport "C:\\Users\\Siqi Wang\\Desktop\\scene\\testOutput\\arch_dining_wall.max" quiet:true

	
	clearSelection()
	select objCheck
	arrayToExport = selection as array
	saveNodes arrayToExport "C:\\Users\\Siqi Wang\\Desktop\\scene\\testOutput\\objCheck_dining_wall.max" quiet:true


	print "done"
)