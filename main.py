import cv2
import os
import glob
import numpy

def drawPred(frame,classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    frame_tmp = frame
    cv2.rectangle(frame_tmp, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    #if classes:
    #   assert(classId < len(classes))
    label = '%s:%s' % (classId, label) #comment out if you have a class_lists.txt with class names in it 
    #label = '%s:%s' % (obj_list[classId], label) #uncomment if you have a class_lists.txt with class names in it 

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    frame_tmp = cv2.rectangle(frame_tmp, (left, int(top - round(1.5*labelSize[1]))), (left + int(round(1.5*labelSize[0])), top + baseLine), (255, 255, 255), cv2.FILLED)
    frame_tmp = cv2.putText(frame_tmp, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
    return frame_tmp

def drawGT(frame, classId, left, top, right, bottom):
    # Draw a bounding box.
    frame_gt = frame
    cv2.rectangle(frame_gt, (left, top), (right, bottom), (255, 178, 50), 3)
    
    #label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    #if classes:
    #   assert(classId < len(classes))
    label = '%s' % (classId) #comment out if you have a class_lists.txt with class names in it 
    #label = '%s' % (obj_list[classId]) #uncomment if you have a class_lists.txt with class names in it

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    frame_gt = cv2.rectangle(frame_gt, (left, int(top - round(1.5*labelSize[1]))), (left + int(round(1.5*labelSize[0])), top + baseLine), (255, 255, 255), cv2.FILLED)
    frame_gt = cv2.putText(frame_gt, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
    return frame_gt

def convert_yolo_coordinates_to_voc(x_c_n, y_c_n, width_n, height_n, img_width, img_height):
  ## remove normalization given the size of the image
  x_c = float(x_c_n) * img_width
  y_c = float(y_c_n) * img_height
  width = float(width_n) * img_width
  height = float(height_n) * img_height
  ## compute half width and half height
  half_width = width / 2
  half_height = height / 2
  ## compute left, top, right, bottom
  ## in the official VOC challenge the top-left pixel in the image has coordinates (1;1)
  left = int(x_c - half_width) + 1
  top = int(y_c - half_height) + 1
  right = int(x_c + half_width) + 1
  bottom = int(y_c + half_height) + 1
  return left, top, right, bottom

def main():
    img_base = "images/"
    os.chdir(img_base)
    imgs = glob.glob("*.jpg")
    filenames = []
    sl = slice(0,-4)
    for img in imgs:
        filenames.append(img[sl])
    #textfilenames = []
    #for name in filenames:
    #    textfilenames.append(textfilename)

    #with open("../class_lists.txt") as f_c: #uncomment if you have a class_lists.txt with class names in it
    #    obj_list = f_c.readlines() #uncomment if you have a class_lists.txt with class names in it
        #remove whitespace characters like `\n` at the end of each line
    #    obj_list = [x.strip() for x in obj_list] #uncomment if you have a class_lists.txt with class names in it

    gt_base = "../ground-truth-txt/"
    det_base = "../detection-results-txt/"
    count = 0;
    for name in filenames:
        textfilename = name+".txt"
        img_path = "../" + img_base+name+".jpg"
        frame_orig = cv2.imread(img_path)
        frame_orig_1 = cv2.imread(img_path)
        gt_path = gt_base + textfilename
        det_path = det_base + textfilename
        img_height, img_width = frame_orig.shape[:2]
        with open(gt_path, "r") as f_gt:
            content_gt = f_gt.readlines()
        content_gt = [x.strip() for x in content_gt]
        for line in content_gt:
            obj_id, x_c_n, y_c_n, width_n, height_n = line.split() #Comment out if co-ordinates not in YOLO format
            left, top, right, bottom = convert_yolo_coordinates_to_voc(x_c_n, y_c_n, width_n, height_n, img_width, img_height) #Comment out if co-ordinates not in YOLO format
            #obj_id, left, top, right, bottom = line.split() #Uncomment if absolute co-ordinates/VOC
            image_gt = drawGT(frame_orig, obj_id, left, top, right, bottom)
        with open(det_path, "r") as f_det:
            content_det = f_det.readlines()
        content_det = [x.strip() for x in content_det]
        for line in content_det:
            obj_id, conf, x_c_n, y_c_n, width_n, height_n = line.split() #Comment out if co-ordinates not in YOLO format
            #obj_id, conf, left, top, right, bottom = line.split() #Uncomment if absolute co-ordinates/VOC
            conf = float(conf)
            left, top, right, bottom = convert_yolo_coordinates_to_voc(x_c_n, y_c_n, width_n, height_n, img_width, img_height) #Comment out if co-ordinates not in YOLO format
            image_det = drawPred(frame_orig_1, obj_id, conf, left, top, right, bottom)
        gt_res_path = "../ground-truth-BB/"+name+".jpg"
        det_res_path = "../detection-results-BB/"+name+".jpg"
        cv2.imwrite(gt_res_path, image_gt)
        cv2.imwrite(det_res_path, image_det)
        count += 1;
        print(str(count)+"/"+str(len(filenames))+" done!")
    print("completed")

if __name__ == "__main__":
    main()