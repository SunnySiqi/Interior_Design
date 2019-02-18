import MaxPlus
vertexDictionary = {}
maxNodeDictionary = {}
nodeDictionary = {}
import sys
#import maya.cmds as cmds
recursionSafety = 20
recursionSafetyCount = 0
dateFilename = ""
import datetime
treeNodesWithVertexData = []
import os
sceneName = ""

class MyTreeNode:
	depth = 0
	vertexData = ""
	name = ""
	parent = None
	children = set()
	
	def __init__(self, name, depth):
		self.name = name
		self.depth = depth 
		self.children = set()
		
	def __init__(self, name, depth, parent, vertexData):
		self.name = name
		self.depth = depth
		self.parent = parent
		self.vertexData = vertexData
		self.children = set()
		
	def addChild(self, node):
		self.children.add(node)
		
	def __str__(self):
		return self.name

def visitNodeForTree(parentTreeNode, maxNode):
	
	maxNodeDictionary[maxNode.Name] = maxNode
	depth = 0
	if parentTreeNode != None:
		depth = parentTreeNode.depth + 1
	
	vertexData = ""
	if maxNode.Name in vertexDictionary.keys():
		vertexData = vertexDictionary[maxNode.Name]
	
	newNode = MyTreeNode(maxNode.Name, depth, parentTreeNode, vertexData)
	nodeDictionary[newNode.name] = newNode #add it to the node dictionary
	count = 0
	newNode.children = set()
	for child in maxNode.Children:
		count = count + 1
		newNode.addChild(visitNodeForTree(newNode, child))
	#print ("return " + newNode.name + " with child count " + str(count) + ", " + str(len(newNode.children))) 
	
	# filters out nodes without vertex data for the matching algorithm
	if maxNode.Name in vertexDictionary.keys():
		treeNodesWithVertexData.append(newNode)
	
	return newNode
	
def printTree(treeNode, printPreface):
	print (str(printPreface) + str(treeNode.name) + " " + str(treeNode.depth) + " vertexData:" + str(treeNode.vertexData))
	for child in treeNode.children:  
		printTree(child, printPreface + "\t")

				
def extractNodeToFirstLevel(treeNode):
	oldParent = treeNode.parent
	#print ("extract " + treeNode.name)
	#print ("old parent: " + oldParent.name)
	if oldParent != None:
		newParent = oldParent 
		# roll all the way up to the first level
		rootName = str(MaxPlus.Core.GetRootNode().Name)
		while newParent.name != rootName:
			newParent = oldParent.parent
		#print ("new parent: " + newParent.name)
		
		#print ("old parent children before: " + str(treeNode in oldParent.children))
		oldParent.children.remove(treeNode)
		#print ("old parent children after: " + str(treeNode in oldParent.children))
		
		#print ("new parent children before: " + str(treeNode in newParent.children))
		newParent.children.add(treeNode)
		#print ("new parent children after: " + str(treeNode in newParent.children))
		
		treeNode.parent = newParent
		#print ("treeNode's parent is now " + treeNode.parent.name)
		
def makeFilepath(newPath, newFile):
	currentFilepath = os.path.realpath(__file__)
	directory = ""

	if("output" in newPath):
		directory = currentFilepath.replace("interiordesign\\scripts\\1_treeMatchingAlgorithmJan29.py", newPath)
		if os.path.exists(directory) == False:
			os.mkdir(directory)
			print ("made directory " + directory)
		directory = currentFilepath.replace("interiordesign\\scripts\\1_treeMatchingAlgorithmJan29.py", newPath + "\\" + newFile)
	else:
		directory = currentFilepath.replace("1_treeMatchingAlgorithmJan29.py", newFile)
	return directory
	

def tempTreeTraversal(treeNode):
	originalChildrenSet = treeNode.children.copy()
	for child in originalChildrenSet:
		if len(originalChildrenSet) == len(treeNode.children):
			if child.name == "default002":
				extractNodeToFirstLevel(child)
				break
			else:
				tempTreeTraversal(child)
		else: 
			break
			
			
def tempTreeTraversalReturn(maxNode, nodeName):
	for child in maxNode.Children:
		if child.Name == nodeName:
			return child
			break
		else:
			returned = tempTreeTraversalReturn(child, nodeName)
			
			return returned
	return None

class NodeMatch:
	nodeSet = set()
	def __init__(self, nodeA, nodeB):
		self.nodeSet = set()
		self.nodeSet.add(nodeA)
		self.nodeSet.add(nodeB)
		
	def __str__(self):
		answer = ""
		for item in self.nodeSet:
			answer += item.name + " "
		return answer

def isLeafNode(node):
	return len(node.children) == 0

def addNodeMatch(matchList, arr):
	for item in arr:
		matchList.append(item)
	return matchList

def treeNodeCompare(node1, node2):
	matchList = []
	
	# if both are leaf nodes, check their vertex data
	if(isLeafNode(node1) and isLeafNode(node2)):
		#if they do have vertex data
		if node1.name in vertexDictionary.keys() and node2.name in vertexDictionary.keys():
			# if they are matching nodes
			if vertexDictionary[node1.name] == vertexDictionary[node2.name]:
				return [NodeMatch(node1, node2)]
	return [] # TODO FIX
	
	#if node1 is a leaf and node2 is not
	if(isLeafNode(node1) and isLeafNode(node2) == False):
		for child in node2.children:
			matchList = addNodeMatch(matchList, treeNodeCompare(node1, child))
		return matchList
		
	# id node1 is not a leaf node and node2 is
	if(isLeafNode(node1) == False and isLeafNode(node2)):
		for child in node1.children:
			matchList = addNodeMatch(matchList, treeNodeCompare(node1, child))
		return matchList
		
	# if neither are leaf nodes	
	if (isLeafNode(node1) == False and isLeafNode(node2) == False):
		for child1 in node1.children:
			for child2 in node2.children:
				matchList = addNodeMatch(matchList, treeNodeCompare(child1, child2))
		return matchList
		
	return matchList


def addNode(stack, node):
	stack.append(node)
	for child in node.children:
		stack = addNode(stack, child)
	return stack

def makeMatchDictionaryKey(node1, node2):
	return node1.name + "-" + node2.name

def parseMatchDictionaryKey(key):
	return key.split("-")
	
def isKeyInMatchDictionary(dictionary, node1, node2):
	key1 = makeMatchDictionaryKey(node1, node2)
	key2 = makeMatchDictionaryKey(node2, node1)
	return key1 in dictionary.keys() or key2 in dictionary.keys()

def isKeyInAlreadyComparedSet(alreadyComparedSet, node1, node2):
	key1 = makeMatchDictionaryKey(node1, node2)
	key2 = makeMatchDictionaryKey(node2, node1)
	return key1 in alreadyComparedSet or key2 in alreadyComparedSet

def getSetToString(set):
	ans = "{"
	for item in set:
		ans += str(item) + " "
	ans += "}"
	return ans

#manipulate the tree and return true if another pass needs to happen, false if not
def assessMatch(key, matchingLeaves):
	
	needToDoAnotherMatchPass = False
	nodeNames = parseMatchDictionaryKey(key)
	node1 = nodeDictionary[nodeNames[0]]
	node2 = nodeDictionary[nodeNames[1]]
	
	#print ("\n#1 assess match node1, node2 " + node1.name + " " + node2.name)
	
	if node1.parent.depth != 0:
		node1 = node1.parent
		
	if node2.parent.depth != 0:
		node2 = node2.parent
	
	node1MatchSet = set()
	node2MatchSet = set()
	
	#print ("#2 assess match node1, node2 " + node1.name + " " + node2.name)

	
	for leafMatch in matchingLeaves:
		for leafNode in leafMatch.nodeSet:
			if leafNode.parent.name == node1.name:
				node1MatchSet.add(leafNode)
			if leafNode.parent.name == node2.name:
				node2MatchSet.add(leafNode)
				
	#print ("\tnode1MatchSet: " + getSetToString(node1MatchSet))
	#print ("\tnode2MatchSet: " + getSetToString(node2MatchSet))
	#print ("\tnode1 children: " + getSetToString(node1.children))
	#print ("\tnode2 children: " + getSetToString(node2.children))
				
	# for all of node1 s children
	for child in node1.children:
		# if that child is not matching with part of node2, extract it from the tree
		#print ("\t\tnode1 child: " + child.name + " in node1MatchSet " + str(child in node1MatchSet))
		if (child in node1MatchSet) == False:
			#print "\t\t\textract it!"
			#extractNodeToFirstLevel(child)
			needToDoAnotherMatchPass = True
			
	for child in node2.children:
		#print ("\t\tnode2 child: " + child.name + " in node2MatchSet " + str(child in node2MatchSet))
		if (child in node2MatchSet) == False:
			#extractNodeToFirstLevel(child)
			#print "\t\t\textract it!"
			needToDoAnotherMatchPass = True
			
	return needToDoAnotherMatchPass
 				
	
def addToDictionarySet(dictionary, key, item):
	if (key in dictionary.keys()) == False:
		dictionary[key] = set()
	
	if (isinstance(item, list)):
		for i in item:
			dictionary[key].add(i)
	else:
		dictionary[key].add(item)
		
	return dictionary
	
def appendMatchSetItems(matchSet, matchSetToAdd):
	for item in matchSetToAdd:
		matchSet.add(item)
	return matchSet
	
def getLeafNodes(node):
	matchSet = set()
	# if it's a leaf node, return itself
	if len(node.children) == 0:
		matchSet.add(node)
		return matchSet
	
	# if its not a leaf node, append all the leaf nodes
	for child in node.children:
		matchSet = appendMatchSetItems(matchSet, getLeafNodes(child))
	
	return matchSet

def getDate():
	date = str(datetime.datetime.now())
	date = date.replace(":", "_")
	date = date.replace(" ", "_")
	date = date.replace(".", "_")
	date = date.replace("-", "_")
	return date
	
def getTotalLeafMatches(parentName, parentDictionary, nodeMatchDictionary):
	parent = nodeDictionary[parentName]
	
	#if parent is a leaf node 
	if len(parentDictionary[parentName]) == 1:
		for childName in parentDictionary[parentName]:
			if childName == parentName:
				return nodeMatchDictionary[childName] # set of matches for that child
				
	matchSet = set()
	for childName in parentDictionary[parentName]:
		if childName != parentName:
			if childName in parentDictionary.keys():
				matchSetToAdd = getTotalLeafMatches(childName, parentDictionary, nodeMatchDictionary)
				matchSet = appendMatchSetItems(matchSet, matchSetToAdd)
			else:
				matchSet = appendMatchSetItems(matchSet, nodeMatchDictionary[childName])
				
	return matchSet
	
	
def newAlgorithm(root):
	stack = list(treeNodesWithVertexData)
	stack2 = list(treeNodesWithVertexData)
	#stack = addNode(stack, root)
	#stack2 = list(stack)
	
	matchDictionary = {}
	recursionSafetyCount = 0

	alreadyComparedSet = set()
	print ("tree nodes with vertex data : " + getSetToString(treeNodesWithVertexData))

	# Match Pass
	for node1 in stack:
		for node2 in stack2:
			# if they arent the same node
			if node1.name != node2.name:
				# if we have not already checked that match
				if isKeyInAlreadyComparedSet(alreadyComparedSet, node1, node2) == False:
				#if isKeyInMatchDictionary(matchDictionary, node1, node2) == False:
					matches = treeNodeCompare(node1, node2)
					alreadyComparedSet.add(makeMatchDictionaryKey(node1, node2))
					
					if len(matches) > 0:
						matchDictionary[makeMatchDictionaryKey(node1, node2)] = matches
	

	# Assess Matches
	nodeMatchDictionary = {}
	parentDictionary = {}
	
	for key in matchDictionary.keys():
		# if there is match content
		if len(matchDictionary[key]) > 0:
			#print (key + ":" + getSetToString(matchDictionary[key]))
			
			# get names of nodes in match from key
			nodeNames = parseMatchDictionaryKey(key)
			node1 = nodeDictionary[nodeNames[0]]
			node2 = nodeDictionary[nodeNames[1]] 
			
			#make the nodeMatchDictionary reference for both nodes
			nodeMatchDictionary = addToDictionarySet(nodeMatchDictionary, node1.name, matchDictionary[key])
			nodeMatchDictionary = addToDictionarySet(nodeMatchDictionary, node2.name, matchDictionary[key])
			
			# parent dictionary: base node1
			if node1.parent.depth == 0:
				parentDictionary = addToDictionarySet(parentDictionary, node1.name, node1.name)
			else:
				# parent dictionary: recure up node1
				n = node1
				while n.parent.depth > 0:
					parentDictionary = addToDictionarySet(parentDictionary, n.parent.name, n.name)
					n = n.parent
					
			# parent dictionary: base node2
			if node2.parent.depth == 0:
				parentDictionary = addToDictionarySet(parentDictionary, node2.name, node2.name)
			else:
				# parent dictionary: recure up node1
				n = node2
				while n.parent.depth > 0:
					parentDictionary = addToDictionarySet(parentDictionary, n.parent.name, n.name)
					n = n.parent
					
	
	#make the  parent match dictionary
	parentMatchDictionary = {}
	for key in parentDictionary.keys():
		#print (key + " is parent of " + getSetToString(parentDictionary[key]))
		parentMatchDictionary[key] = getTotalLeafMatches(key, parentDictionary, nodeMatchDictionary)
		#print getSetToString(parentMatchDictionary[key])
		
	#print "\ndouble key loop"
	
	
	
	# CURRENT PROBLEM, SETS ELIMINATE DUPLICATE MATCHES 
	#comparing Group2098672384 and WRLO1179
	#parentMatchDict[key1] {Object4829 Object4836  Box5761 Box5762  THRE3807 THRE3806  Object4833 Object4840  Object4837 Object4830  Object4834 Object4835  Object4832 Object4839  Object4838 Object4831  }
	#parentMatchDict[key2] {Object4837 Object4830  Object4829 Object4836  Box5761 Box5762  THRE3807 THRE3806  Object4838 Object4831  Object4833 Object4840  Object4834 Object4835  Object4832 Object4839  }
	#subtraction #1: {}
	#subtraction #2: {}
	#Group2098672384's leaf nodes: {Object4829 Box5761 Object4833 Object4831 Object4830 THRE3806 Object4834 Object4832 }
	#WRLO1179's leaf nodes: {Object4829 Object4831 Object4835 THRE3807 Object4839 Object4834 Box5762 Object4832 Object4833 Object4838 Object4840 Box5761 Object4837 THRE3806 Object4830 Object4836 }
	
	instancesString = []
	accessMatchViaOne = {}
	setToDetach = set()
	
	# pair the matchDictionary items
	matchParentMatchDictionary = {}
	for key1 in parentMatchDictionary.keys():
		for key2 in parentMatchDictionary.keys():
			# if the keys are not the same
			if key1 != key2:
				
				#print ("\ncomparing " + key1 + " and " + key2)
				
				# if the sets have some common matches
				if len(parentMatchDictionary[key1] & parentMatchDictionary[key2]) > 0:
#				len(parentMatchDictionary[key1]) or len(parentMatchDictionary[key2] - parentMatchDictionary[key1]) < len(parentMatchDictionary[key2]):
					# get the nodes
					node1 = nodeDictionary[key1]
					node2 = nodeDictionary[key2]

					
					#print ("\tcomparing " + key1 + " and " + key2)
					#print ("\tparentMatchDict[key1] " + getSetToString(parentMatchDictionary[key1]))
					#print ("\tparentMatchDict[key2] " + getSetToString(parentMatchDictionary[key2]))
					
					leafNodes = getLeafNodes(node1)
				#	print ("\t" + node1.name + "'s leaf nodes: " + getSetToString(leafNodes))
					
					leafNodes2 = getLeafNodes(node2)
				#	print ("\t" + node2.name + "'s leaf nodes: " + getSetToString(leafNodes2))
					
					matchesInCommonWithNodes = parentMatchDictionary[key1] & parentMatchDictionary[key2]
				#	print ("\tintersection set: " + getSetToString(matchesInCommonWithNodes))
					
					matchesInCommonNodeSet = set()
					for match in matchesInCommonWithNodes:
						for n in match.nodeSet:
							matchesInCommonNodeSet.add(n)
						
				#	print ("\tmatchesInCommonNodeSet: " + getSetToString(matchesInCommonNodeSet)) 
					
					# Check if any of Node1's real leaf nodes are not included in this match
					#print "\tLeafnode assessment #1"

					n1Parent = node1
					n2Parent = node2
					#print ("n1parent " + n1Parent.name + " " + str(n1Parent.depth))

					while n1Parent.depth != 1 and n1Parent.parent != None:
						n1Parent = n1Parent.parent
						#print ("n1parent " + n1Parent.name + " " + str(n1Parent.depth))


					#print ("n2parent " + n2Parent.name + " " + str(n2Parent.depth))
					while n2Parent.depth != 1 and n2Parent.parent != None:
						n2Parent = n2Parent.parent
						#print ("n2parent " + n2Parent.name + " " + str(n2Parent.depth))

					if n1Parent.name == n2Parent.name:
						#print ("coninuing for " + key1 + " " + key2)
						#print ("\tparents" + n1Parent.name + " " + n2Parent.name)
						continue

					count = 0
					collectCommonNodeMatchNames = set()

					for leafNode in leafNodes:
						if (leafNode in matchesInCommonNodeSet):
							count += 1
							collectCommonNodeMatchNames.add(leafNode.name)
							#print ("\t\t" + leafNode.name + " is all set :)")
							
					if count == len(leafNodes):
						accessMatchViaOne = addToDictionarySet(accessMatchViaOne, node1.name, node2.name)
					else:
						print ("comparing " + key1 + " and " + key2)
						#detach the matched nodes if it wasnt a complete group match
						for nodeName in collectCommonNodeMatchNames:
							n = nodeDictionary[nodeName]
							nParent = n.parent
							print ("going to add the parent of " + n.name)
							if nParent.name == node1.name:
								setToDetach.add(n.name)
								print ("added " + n.name)
							else:
								while nParent.parent.name != node1.name:
									nParent = nParent.parent
								setToDetach.add(nParent.name)
								print ("added " + nParent.name)

						print "COUNT IS NOT EQUAL TO THE LEAFNODES"	
						print (node1.name + " " + node2.name)	
					#print "\tLeafnode assessment #2"
					count = 0
					collectCommonNodeMatchNames = set()

					for leafNode in leafNodes2:
						if (leafNode in matchesInCommonNodeSet):
							count += 1
							collectCommonNodeMatchNames.add(leafNode.name)
							#print ("\t\t" + leafNode.name + " is all set :)")
							
					if count == len(leafNodes2):
						accessMatchViaOne = addToDictionarySet(accessMatchViaOne, node2.name, node1.name)
					else:
						#detach the matched nodes if it wasnt a complete group match
						for nodeName in collectCommonNodeMatchNames:
							n = nodeDictionary[nodeName]
							nParent = n.parent
							print ("going to add the parent of " + n.name)

							if nParent.name == node2.name:
								setToDetach.add(n.name)
								print ("added " + n.name)

							else:
								while nParent.parent.name != node2.name:
									nParent = nParent.parent
								setToDetach.add(nParent.name)
								print ("added " + nParent.name)

						print "COUNT IS NOT EQUAL TO THE LEAFNODES"	
						print (node1.name + " " + node2.name)
						
	#for key in accessMatchViaOne.keys():
		#print ("access match- key: " + key + " set: " + getSetToString(accessMatchViaOne[key]))
	#print "chair analysis:"
	#print accessMatchViaOne['LARK8583']
		
	takenCareOf = set()
	setToExport = set()
	setToExportDictionary = {}
	for key1 in parentMatchDictionary.keys():
		for key2 in parentMatchDictionary.keys():
			# if the keys are not the same
			if key1 != key2:
				if key1 in accessMatchViaOne.keys() and key2 in accessMatchViaOne.keys():
					if key1 in accessMatchViaOne[key2] and key2 in accessMatchViaOne[key1]:
						#print ("match: " + key1 + " " + key2)
						# neither have been looked at
						if (key1 in takenCareOf) == False and (key2 in takenCareOf) == False:
							setToExport.add(key1)
							takenCareOf.add(key1)
							takenCareOf.add(key2)
							setToExportDictionary = addToDictionarySet(setToExportDictionary, key1, key2)
							continue
						# key1 has been taken care of but not key2, so just add it to the set key1 is in
						if (key1 in takenCareOf) and (key2 in takenCareOf) == False:
							if key1 in setToExport:
								takenCareOf.add(key2)
								setToExportDictionary = addToDictionarySet(setToExportDictionary, key1, key2)
								continue
						# key1 is not taken care of but key2 is
						if (key1 in takenCareOf) == False and key2 in takenCareOf:
							if key2 in setToExport:
								takenCareOf.add(key1)
								setToExportDictionary = addToDictionarySet(setToExportDictionary, key2, key1)

					#	if (key1 in setToExport) == False and (key2 in setToExport) == False:
							#setToExport.add(key1)
						#	takenCareOf.add(key1)
						#	takenCareOf.add(key2)
						#	setToExportDictionary = addToDictionarySet(setToExportDictionary, key1, key2)
						#else:
						#	if key1 in setToExport:
						#		setToExportDictionary = addToDictionarySet(setToExportDictionary, key1, key2)
						#	if key2 in setToExport:
						#		setToExportDictionary = addToDictionarySet(setToExportDictionary, key2, key1)
	
	#print ("set to export: " + getSetToString(setToExport))

	#newSetToExport = set()
	keysToRemove = set()

	for k in setToExportDictionary.keys():
		print (k + ": " + getSetToString(setToExportDictionary[k]))

	for key1 in setToExport:
		for key2 in setToExport:
			if key1 != key2:
				node1 = nodeDictionary[key1]
				node2 = nodeDictionary[key2]
				if node1 in node2.children:
					keysToRemove.add(key1) #node1 is already going to be a part of node2
				if node2 in node1.children:
					keysToRemove.add(key2)

	print ("keys to delete: " + getSetToString(keysToRemove))

	for keyToDelete in keysToRemove:
		setToExport.remove(keyToDelete)
		del setToExportDictionary[keyToDelete]
	
	for k in setToExportDictionary.keys():
		print (k + ": " + getSetToString(setToExportDictionary[k]))


	#print ("takenCareOf: " + getSetToString(takenCareOf))
	nonMatchesToExport = set()
	for n in treeNodesWithVertexData:
		#if (n.name in accessMatchViaOne.keys()) == False:
		if (n.name in takenCareOf) == False:
			nodeToAdd = n
			#print ("original node to add: " + n.name)
			while nodeToAdd.parent.depth > 0:
				nodeToAdd = nodeToAdd.parent
			if (nodeToAdd.name in takenCareOf) == False:
				nonMatchesToExport.add(nodeToAdd.name)
			

	print ("nonMatchesToExport " + getSetToString(nonMatchesToExport))
	
	date = getDate()

	toExportFilepath = makeFilepath("output\\toExportData", "")

	toExportFilename = toExportFilepath + "\\" + date + "-" + sceneName + ".txt"
	instancesToExportFile = open(toExportFilename, "w")
	for item in setToExport:
		text = item + ","
		for subMatch in setToExportDictionary[item]:
			text += subMatch + ","
			
		text = text[:len(text) - 1]
		#print ("text: " + text)
		instancesToExportFile.write(text + "\n")
	for item in nonMatchesToExport:
		instancesToExportFile.write(str(item) + "\n")

	instancesToExportFile.close()
	print ("exported instance data to " + toExportFilename)

	toDetachFilename = toExportFilepath + "\\" + date + "-" + sceneName + "_NodesToDetach.txt"
	toDetachFile = open(toDetachFilename, "w")
	for item in setToDetach:
		text = item + "\n"
		print text
		toDetachFile.write(text)
	toDetachFile.close()




	dataFilepathsFilepath = makeFilepath("", "dataFilepaths.txt")
	dataFilepathsFile = open(dataFilepathsFilepath, "w")
	dataFilepathsFile.write(toExportFilename + "\n")
	dataFilepathsFile.write(sceneName + "\n")
	dataFilepathsFile.close()
	
	
	
	
	
	#instancesToExportFile.close()
						
					# ALSO TODO THIS ISNT WORKING FOR DEFAULTS
					#for child in leafNodes:
						#found = False
						#for match in parentMatchDictionary[key1]:
							#if child in match.nodeSet:
								#found = True
						#if found == False:
							#print ("\t" + "didnt find a match for " + child.name + " in parentMatchDictionary for " + key1 + " dict: " + getSetToString(parentMatchDictionary[key1]))
							
					#for child in leafNodes2:
						#found = False
						#for match in parentMatchDictionary[key2]:
							#if child in match.nodeSet:
								#found = True
						#if found == False:
							#print ("\t" + "didnt find a match for " + child.name + " in parentMatchDictionary for " + key1 + " dict: " + getSetToString(parentMatchDictionary[key1]))
				

def readVertexData():
	# get the filepath of the vertex data from script #0
	dataFilepathsFilepath = makeFilepath("", "dataFilepaths.txt")
	print dataFilepathsFilepath
	dataFilepathsFile = open(dataFilepathsFilepath, "r")
	directory = dataFilepathsFile.readline()
	sceneName = dataFilepathsFile.readline()
	dataFilepathsFile.close()
	
	# clean out the new line characters so the filepaths will work
	directory = directory.replace("\n", "")
	sceneName = sceneName.replace("\n", "")

	# read the vertex data into the dictionary
	f = open(directory, "r")
	linesString = f.read()
	lines = linesString.split("\n")
	for line in lines:
		arr = line.split(":")
		if len(arr) > 1:
			vertexDictionary[arr[0]] = arr[1]
	f.close()

	# return the scene name so it can be used later on
	return sceneName
	
def visitNode(node):
	nodeDictionary = {}
	for child in node.Children:
		childDictionary = visitNode(child)
		nodeDictionary[child.Name] = childDictionary
	return nodeDictionary
	
if __name__ == '__main__':

	sceneName = readVertexData()
	rootNode = visitNodeForTree(None, MaxPlus.Core.GetRootNode())
	newAlgorithm(rootNode)
	#MaxPlus.SelectionManager.ClearNodeSelection()
	#print "selection cleared"
	#MaxPlus.SelectionManager.SelectNode(node)
	#print (node.Name + " selected"

	print "done with script #1 tree matching\n"

	nextScript = makeFilepath("", "2_exportInstancesJan29.ms")
	#nextScript = nextScript.replace("C:", "")
	print ("nextScript: " + nextScript)
	try:
		result = MaxPlus.Core.EvalMAXScript("fileIn \"" + nextScript + "\"")
		print ("result of calling next script: " + result)
	except:
		print "exception may have occured when calling script 2 from script 1 "
