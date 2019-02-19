(
	allObjects = objects as array
	allTypes = #()

/*
	for obj in allObjects do (
		--print obj.name as string
		--print obj.baseObject as string
		appendIfUnique allTypes obj.baseObject
	)
	
	makeUniqueArray allTypes
	for t in allTypes do(
		print t as string
	)
*/
	for obj in allObjects do(
		print ("Object:" + obj.name + " classOf:" + (classOf obj as string) + " Baseobject:" + (obj.baseObject as string))
	)
	print "done"
)