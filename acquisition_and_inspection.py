import os
import PySpin
import preprocess
import acquisition
import cv2
from pathlib import Path
from multi_comparison import MultiComparison
from multi_inspection import MultiInspector
from multi_inspection_result import MultiInspectionResult
from novelty_detector import NoveltyDetector
from project import Dataset, Project

def acquire_and_inspect():

    tag = '0'

    path = input('Enter project path (q:quit):')
    if path == 'q':
        return
    project = Project(Path(path))
    inspector = MultiInspector(project=project, target_area_tags=sorted(project.all_area_tags(directory_name='preprocessed')))
    inspector.load_train_ok_dists()
    
    print('Inspection starts.')

    alpha = input('alpha value (default: 0.0013):')
    try:
        alpha = float(alpha)
    except:
        alpha = 0.013

    counter = int(input('Enter start number:'))


    try:
        test_file = open('test.txt', 'w+')
    except IOError:
        print('Unable to write to current directory. Please check permissions.')
        input('Press Enter to exit...')
        return False

    test_file.close()
    os.remove(test_file.name)

    result = True

    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter')
        return False


    while True:

        # dt_now = datetime.now()
        # filename = str(dt_now.date())+'-'+str(dt_now.hour)+str(dt_now.minute)+str(dt_now.second)+'_'+tag+'.jpg'
        filename = str(counter)+'_'+tag+'.jpg'
        filename_normalized = str(counter)+'_n_'+tag+'.jpg'
        q = input('Press Enter to aquire image (q:quit):')
        if q == 'q':
            break
        if not os.path.exists(path+'/inspection_targets/'+str(counter)):
            os.makedirs(path+'/inspection_targets/'+str(counter))
        filepath = path+'/inspection_targets/'+str(counter)+'/'+filename
        for i, cam in enumerate(cam_list):
            print('Running example for camera %d...' % i)
            result &= acquisition.run_single_camera(cam, filepath)
            print('Camera %d example complete... \n' % i)
            if not result:
                break
        if not os.path.exists(path+'/inspection_targets/'+str(counter))+'_n':
            os.makedirs(path+'/inspection_targets/'+str(counter)+'_n')
        image = cv2.imread(filepath)
        image_normalized = preprocess.crop_resize_normalize(image)
        cv2.imwrite(path+'/inspection_targets/'+str(counter)+'_n/'+filename_normalized, image_normalized)
        
        # capture_id = input('capture id (q:quit):')
        # if capture_id == 'q':
        #     break 
        # elif not capture_id in os.listdir(path+'/inspection_targets'):
        #     print('This capture id is not in the inspection_targets directory.')
        #     continue

        capture_id = str(counter)+'_n'

        inspection_result = inspector.multi_inspect(capture_id=str(capture_id), alpha=alpha)
        print(inspection_result.is_positive())

        print('Done!')
        counter += 1

if __name__ == "__main__":
    acquire_and_inspect()