(
	PreObjects = objects as array
	for obj in PreObjects do (
		if obj.parent == undefined then(
			print ("Parent Node   " + obj.name as string  + "   " + obj.baseObject as string )
		)
		else
			print ("Children Node   " + obj.name as string  + "   " + obj.baseObject as string )
	)
)