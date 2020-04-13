import urllib.request
import cv2
import os

def store_raw_images(neg_images_link, pic_start, pic_max):
    
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
    
    pic_num = pic_start
    
    if not os.path.exists('neg'):
        os.makedirs('neg')

    n = len(neg_image_urls.split('\n'))
    m = 0
        
    for i in neg_image_urls.split('\n'):
        print(m, "/", n, end=" ")
        m += 1
        try:
            print("SUCESS", pic_num, "/", pic_max)
            print(i)
            urllib.request.urlretrieve(i, "neg/"+str(pic_num)+".jpg")
            img = cv2.imread("neg/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
            
        except Exception as e:
            print("ERROR")
            print(str(e))
        
        if pic_num >= pic_max:
            break

#http://forestlife.info/Photo/320/45.jpg 391.jpg

people = 'http://imagenet.stanford.edu/api/text/imagenet.synset.geturls?wnid=n00007846'
constructions = 'http://imagenet.stanford.edu/api/text/imagenet.synset.geturls?wnid=n04341686'
terrain = 'http://imagenet.stanford.edu/api/text/imagenet.synset.geturls?wnid=n09287968'
plants = 'http://imagenet.stanford.edu/api/text/imagenet.synset.geturls?wnid=n00017222'

#store_raw_images(people, 392, 2392)
#store_raw_images(constructions, 1207, 4000)
#store_raw_images(plants, 2169, 4000)

















