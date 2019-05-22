import csv
import xml.etree.ElementTree as ET
import os



def readFileNames(dir):
    fileList = []
    for file in os.listdir(dir):
        path = os.path.join(dir+"/", file)
        fileList.append(path)
    return fileList


def parseXML(xmlfile, objList):
    # create element tree object
    tree = ET.parse(xmlfile)

    # get root element
    root = tree.getroot()

    # create empty list for news items
    items = []

    # iterate news items
    for item in root.findall('./object/name'):

        # empty news dictionary
        # news = {}
        #
        # # iterate child elements of item
        # for child in item:
        #
        #     # special checking for namespace object content:media
        #     if child.tag == '{http://search.yahoo.com/mrss/}content':
        #         news['media'] = child.attrib['url']
        #     else:
        #         news[child.tag] = child.text.encode('utf8')
        #
        #         # append news dictionary to news items list
        # newsitems.append(news)
        #
        # # return news items list
        string = item.text
        ret = string.split('\n')[1]
        ret = ret.replace('crop', '')
        ret = ret.replace('occluded', '')
        ret = ret.strip()
        if ret not in objList:
            objList.append(ret)
    return objList


def savetoCSV(newsitems, filename):
    # specifying the fields for csv file
    fields = ['guid', 'title', 'pubDate', 'description', 'link', 'media']

    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(newsitems)


def main():

    # parse xml file
    fileList = readFileNames('D:/Siqi_projects/Co-occurance/data')
    objList = []
    for room in fileList:
        roomXMLs = readFileNames(room)
        for file in roomXMLs:
            objList = parseXML(file, objList)
    print(objList)
    # store news items in a csv file
    #savetoCSV(newsitems, 'topnews.csv')
    with open('objects2.txt', 'w', encoding='utf-8') as f:
        for item in objList:
            f.write("%s\n" % item)


if __name__ == "__main__":
    # calling main function
    main()