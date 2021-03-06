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
/*
example of 2D surface touch --> window in the wall
	pt_2d = point2 pt.x pt.y
	pln_box2 = box2 (point2 pln.max.x pln.max.y) (point2 pln.min.x pln.min.y)
	return contains pln_box2 pt_2d
*/
--smaller obj is bp1

fn checkContain Bp1 Bp2 = (
	obj1_3d = box3 Bp1[1] Bp1[7] 
	obj2_3d = box3 Bp2[1] Bp2[7] 
	return contains obj2_3d obj1_3d
)

fn collisionDetection obj1 obj2 = (
	local foundIntersection = false

	for child1 in obj1.children do (
		for child2 in obj2.children do (
			if intersects child1 child2 do (
				foundIntersection = (child1.mesh * child2.mesh).numverts > 0
				break
			)	
		)
	)

	/*if intersects obj1 obj2 then 
		(obj1.mesh * obj2.mesh).numverts > 0
	else
		false*/
	foundIntersection
)


(
	PreObjects = objects as array
	allObjects = #()
	dummyStuff = #()

	local previousObject = getNodeByName "AM173_054"

	for obj in PreObjects do (
		if obj.parent == undefined do(


			if collisionDetection previousObject obj do print ("collision " + obj.name as string)
			
			objNameAsString = obj.name as string
			if findString objNameAsString "WRLO1179" != undefined do (

			print "found WRLO1179"
			obj_type = obj.baseObject as string
			print ("0bj type " + obj_type)
			boundingBoxPoints = getBBoxPoints obj
			size = getSize boundingBoxPoints
			print boundingBoxPoints
			print size

			
			)
			obj_type = obj.baseObject as string

			if obj_type != "Dummy" do (
				if findString objNameAsString "THRE3804" != undefined do print "found non-dummy THRE3804"
			)
			
			if obj_type == "Dummy" do(
				--print ("Dummy Object !!!!!!!!!!!!!!" + obj.name as string)
				mark = 0
				boundingBoxPoints = getBBoxPoints obj
				size = getSize boundingBoxPoints
				if size[1] > 70 do
					mark += 1
				if size[2] > 70 do
					mark += 1
				if size[3] > 70 do
					mark += 1
				if mark > 1 then (
					if findString objNameAsString "WRLO1179" != undefined do print (obj.name as string + " " + obj.baseObject as string)
					append dummyStuff obj
					for child in obj.children do append dummyStuff child
					continue
				)
				else(
					append allObjects obj
				--	print ("Add this dummy object to the allObject!!!!!!!!!!" + obj.name as string)
				)
					
			)
			index = findString obj_type "VRay"
			if index == undefined do 
				append allObjects obj
		) 
	)

	arch = #()
	objCheck1 = #()
		

	for obj in allObjects do (
		boundingBoxPoints = getBBoxPoints obj
		size = getSize boundingBoxPoints

		if size[1] >200 or size[2] > 200 or size[3] > 200 do (
			--print obj.name
			append arch obj
			continue
		)
		
		append objCheck1 obj
		--print ("Add this object to the list ################" + obj.name as string)
		objNameAsString = obj.name as string
		if findString objNameAsString "WRLO1179" != undefined do print "found WRL in all objects"
		
	)



	clearSelection()
	select arch
	arrayToExport = selection as array
	saveNodes arrayToExport "C:\\Users\\Siqi Wang\\Desktop\\scene\\testOutput\\arch_living.max" quiet:true

	clearSelection()
	select dummyStuff
	arrayToExport = selection as array
	print "\ndummystuff:"
	print dummyStuff
	saveNodes arrayToExport "C:\\Users\\Siqi Wang\\Desktop\\scene\\testOutput\\dummyStuff_living.max" quiet:true
	
	clearSelection()
	select objCheck1
	arrayToExport = selection as array
	saveNodes arrayToExport "C:\\Users\\Siqi Wang\\Desktop\\scene\\testOutput\\objCheck1_living.max" quiet:true



	
		--meshObj = obj.mesh
		--if obj.numverts != undefined do print (obj.name as string + " " + meshObj.numverts as string)

	print "done"
)