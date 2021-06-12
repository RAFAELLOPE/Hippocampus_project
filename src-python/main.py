import numpy as np
from utils.utils import LoadHippocampusData
from trainer.training import UNetExperiment
from trainer.inference import UNetInferenceAgent
from configuration import Config
from radiomics.featureextractor import RadiomicsFeatureExtractor
import SimpleITK as sitk
import pandas as pd
import os, sys
import datetime
import numpy as np
import pydicom
import nibabel as nib
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw




def load_dicom_volume_as_numpy_from_list(dcmlist):
    """Loads a list of PyDicom objects a Numpy array.
    Assumes that only one series is in the array

    Arguments:
        dcmlist {list of PyDicom objects} -- path to directory

    Returns:
        tuple of (3D volume, header of the 1st image)
    """
    slices = [np.flip(dcm.pixel_array).T for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)]
    hdr = dcmlist[0]
    hdr.PixelData = None
    return (np.stack(slices, 2), hdr)


def load_nifti_volme_as_numpy_from_file(nifti_file):
    nib_file = nib.load(nifti_file)
    vol = nib_file.get_fdata()
    hdr = nib_file.header
    return (vol, hdr)



def get_predicted_volumes(pred):
    """Gets volumes of two hippocampal structures from the predicted array

    Arguments:
        pred {Numpy array} -- array with labels. Assuming 0 is bg, 1 is anterior, 2 is posterior

    Returns:
        A dictionary with respective volumes
    """
    anterior = pred[pred == 1]
    volume_ant = np.sum(anterior)
    posterior = pred[pred == 2]
    posterior[posterior == 2] = 1
    volume_post = np.sum(posterior)
    total_volume = volume_post +  volume_ant
    return {"anterior": volume_ant, "posterior": volume_post, "total": total_volume}


def get_radiomic_features(volume, pred_label):
    """Gets volumes of two hippocampal structures from the predicted array

    Arguments:
        volume {Numpy array} -- array with original volumen
        pred_label {Numpy array} -- array with labels predicted. Assuming 0 is bg, 1 is anterior, 2 is posterior

    Returns:
        A dictionary with respective radiomic features for both anterior and posterior.
    """
    settings = {'removeOutliers': 3,
                'normalize':True,
                'label':1,
                'weightingNorm':'euclidean'}
    
    pred_volume = pred_label[:,:volume.shape[1], :volume.shape[2]]

    vol_sitk = sitk.GetImageFromArray(volume)
    pred_sitk = sitk.GetImageFromArray(pred_volume)
    extractor_label_1 = RadiomicsFeatureExtractor(additionalInfo=False, **settings)
    radiomic_features_1 = extractor_label_1.execute(vol_sitk, pred_sitk)

    settings['label'] = 2
    extractor_label_2 = RadiomicsFeatureExtractor(additionalInfo=False, **settings)
    radiomic_features_2 = extractor_label_2.execute(vol_sitk, pred_sitk)

    return {'Anterior' : radiomic_features_1, 'Posterior': radiomic_features_2}


def convert2rgb(img_arr):
    img_clip = np.clip(img_arr, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_clip)
    pil_img = pil_img.convert('RGB')
    img_rgb = np.asarray(pil_img).copy()
    return img_rgb


def overlay_images(background, seg):
    channels = list()
    cond_anterior = np.where(seg == 127)
    cond_posterior = np.where(seg == 255)
    cond = np.where((seg==127) | (seg==255))

    for ch in range(background.shape[2]):
        img_ch = background[...,ch]
        if ch == 0:
            img_ch[cond_anterior] = 255
        elif ch == 1:
            img_ch[cond_posterior] = 255
        else:
            img_ch[cond] = 0
        img_ch = np.expand_dims(img_ch, axis=2)
        channels.append(img_ch)
    
    img_rgb = np.concatenate(tuple(channels), axis=2)
    #img_pil = Image.fromarray(img_rgb)
    return img_rgb


def create_report(inference, orig_vol, pred_vol):
    """Generates an image with inference report

    Arguments:
        inference {Dictionary} -- dict containing anterior, posterior and full volume values
        header {PyDicom Dataset} -- DICOM header
        orig_vol {Numpy array} -- original volume
        pred_vol {Numpy array} -- predicted label

    Returns:
        PIL image
    """
    images = []
    for i_slc in range(volume.shape[0]):
        pimg = Image.new("RGB", (1000, 1000))
        draw = ImageDraw.Draw(pimg)
        #header_font = ImageFont.truetype("arial.ttf", size=40)
        header_font = ImageFont.load_default()
        #main_font = ImageFont.truetype("arial.ttf", size=30)
        main_font = ImageFont.load_default()
        draw.text((10, 0), "HippoVolume.AI", (255, 255, 255), font=header_font)
        draw.multiline_text((10, 90), 
        f"Total hippocampus volume {inference['total']} mm3\nAnterior volume {inference['anterior']} mm3\nPosterior volume {inference['posterior']} mm3", 
        (255, 255, 255), font=main_font)
        # Create a PIL image from array:
        # Numpy array needs to flipped, transposed and normalized to a matrix of values in the range of [0..255]
        org_vol_slc = orig_vol[i_slc,...]
        pred_vol_slc = pred_vol[i_slc,...]
        pred_vol_slc = pred_vol_slc[:org_vol_slc.shape[0], :org_vol_slc.shape[1]]
        nd_img = np.flip((org_vol_slc/np.max(org_vol_slc))*0xff).T.astype(np.uint8)
        nd_msk = np.flip((pred_vol_slc/np.max(pred_vol_slc))*0xff).T.astype(np.uint8)
        rgb_img = convert2rgb(nd_img)
        pil_img = overlay_images(rgb_img, nd_msk)
        pil_resized = Image.fromarray(np.resize(pil_img, rgb_img.shape))
        pil_final = pil_resized.resize((1000-10, 1000-280))        
        # Paste the PIL image into our main report image object (pimg)
        pimg.paste(pil_final, box=(10, 280))
        images.append(pimg)
    return images


def save_report_as_dcm(header, report, path, instance_number, tot_images=32):
    """Writes the supplied image as a DICOM Secondary Capture file

    Arguments:
        header {PyDicom Dataset} -- original DICOM file header
        report {PIL image} -- image representing the report
        path {Where to save the report}

    Returns:
        N/A
    """
    out = pydicom.Dataset()
    out.file_meta = pydicom.Dataset()
    out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    out.is_little_endian = True
    out.is_implicit_VR = False
    # We need to change class to Secondary Capture
    out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    out.file_meta.MediaStorageSOPClassUID = out.SOPClassUID
    # Our report is a separate image series of one image
    out.SeriesInstanceUID = header.SeriesInstanceUID
    out.SOPInstanceUID = pydicom.uid.generate_uid()
    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.Modality = "OT" # Other
    out.SeriesDescription = "HippoVolume.AI"
    out.Rows = report.height
    out.Columns = report.width
    out.ImageType = r"DERIVED\PRIMARY\AXIAL" # We are deriving this image from patient data
    out.SamplesPerPixel = 3 # we are building an RGB image.
    out.PhotometricInterpretation = "RGB"
    out.PlanarConfiguration = 0 # means that bytes encode pixels as R1G1B1R2G2B2... as opposed to R1R2R3...G1G2G3...
    out.BitsAllocated = 8 # we are using 8 bits/pixel
    out.BitsStored = 8
    out.HighBit = 7
    out.PixelRepresentation = 0
    out.InstanceNumber = str(instance_number)
    #out.ImagePositionPatient[-1] = str(instance_number)
    out.ImagePositionPatient = ['0', '0', str(header['qoffset_z'])]
    # Set time and date
    dt = datetime.date.today().strftime("%Y%m%d")
    tm = datetime.datetime.now().strftime("%H%M%S")
    out.StudyDate = dt
    out.StudyTime = tm
    out.SeriesDate = dt
    out.SeriesTime = tm
    out.ImagesInAcquisition = str(tot_images)
    # We empty these since most viewers will then default to auto W/L
    out.WindowCenter = ""
    out.WindowWidth = ""
    # Data imprinted directly into image pixels is called "burned in annotation"
    out.BurnedInAnnotation = "YES"
    out.PixelData = report.tobytes()
    pydicom.filewriter.dcmwrite(path, out, write_like_original=False)


def get_series_for_inference(path):
    """Reads multiple series from one folder and picks the one
    to run inference on.

    Arguments:
        path {string} -- location of the DICOM files

    Returns:
        Numpy array representing the series
    """
    dicoms = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path)]

    #vol = nib.load(path).get_fdata()

    series_for_inference = [ds for ds in dicoms if ds.SeriesDescription == 'HippoCrop']
    #print(len(series_for_inference))
    # Check if there are more than one series (using set comprehension).
    if len({f.SeriesInstanceUID for f in series_for_inference}) != 1:
        print("Error: can not figure out what series to run inference on")
        return []

    return series_for_inference

def os_command(command):
    # Comment this if running under Windows
    #sp = subprocess.Popen(["/bin/bash", "-i", "-c", command])
    #sp.communicate()
    os.system(command)


""" if __name__ == "__main__":
    app.run(debug=True, port=6969) """
    


if __name__ == "__main__":
    # This code expects a single command line argument with link to the directory containing
    # routed studies

    # Find all subdirectories within the supplied directory. We assume that 
    # one subdirectory contains a full study
    # Define paths
    base_path = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(base_path, "..", "data"))
    model_path = os.path.abspath(os.path.join(base_path, "..", "data", "model_weigths", "model.pth"))
    uploads_dir = os.path.abspath(os.path.join(base_path, "..", "data", "uploads"))
    results_dir = os.path.abspath(os.path.join(base_path, "..", "data", "results"))

    files_dir = [os.path.join(uploads_dir, d) for d in os.listdir(uploads_dir) if  os.path.join(uploads_dir, d).endswith('.nii.gz')]

    # Load model inference agent 
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    inference_agent = UNetInferenceAgent(device=device,
                                         parameter_file_path= model_path)

    for f in files_dir:
        #volume, header = load_dicom_volume_as_numpy_from_list(get_series_for_inference(f))
        volume, header = load_nifti_volme_as_numpy_from_file(f)
        
        # Run inference
        pred_label = inference_agent.single_volume_inference_unpadded(volume)
        pred_volumes = get_predicted_volumes(pred_label)
        radiomic_features = get_radiomic_features(volume, pred_label)
        df_rad_features = pd.DataFrame(radiomic_features)

        # Create and save the report
        header.SeriesInstanceUID = pydicom.uid.generate_uid()
        print("Creating and pushing report...")

        report_imgs = create_report(pred_volumes, volume, pred_label)
        for i, report_img in enumerate(report_imgs):
            report_path = os.path.join(results_dir, f"report_{i+1}.dcm")
            save_report_as_dcm(header, report_img, report_path, i+1)
        
        df_rad_features.to_csv(os.path.join(results_dir,'rad_features.csv'))
    

""" if __name__ == "__main__":
    # This code expects a single command line argument with link to the directory containing
    # routed studies
    if len(sys.argv) != 4:
        print("You should supply one command line argument pointing to the routing folder. Exiting.")
        sys.exit()

    # Find all subdirectories within the supplied directory. We assume that 
    # one subdirectory contains a full study
    subdirs = [os.path.join(sys.argv[1], d) for d in os.listdir(sys.argv[1]) if os.path.isdir(os.path.join(sys.argv[1], d))]

    # Get the latest directory
    study_dir = sorted(subdirs, key=lambda dir: os.stat(dir).st_mtime, reverse=True)[0]
    print(f"Looking for series to run inference on in directory {study_dir}...")
    volume, header = load_dicom_volume_as_numpy_from_list(get_series_for_inference(study_dir))
    print(f"Found series of {volume.shape[2]} axial slices")
    print("HippoVolume.AI: Running inference...")
    model_path = sys.argv[2]
    inference_agent = UNetInferenceAgent(device="cpu",
                                         parameter_file_path= model_path)

    # Run inference
    pred_label = inference_agent.single_volume_inference_unpadded(np.array(volume))
    pred_volumes = get_predicted_volumes(pred_label)
    radiomic_features = get_radiomic_features(volume, pred_label)
    df_rad_features = pd.DataFrame(radiomic_features)
    # Create and save the report
    header.SeriesInstanceUID = pydicom.uid.generate_uid()
    print("Creating and pushing report...")
    report_save_path = sys.argv[3]
    report_imgs = create_report(pred_volumes, header, volume, pred_label)
    for i, report_img in enumerate(report_imgs):
        report_path = os.path.join(report_save_path, f"report_{i+1}.dcm")
        save_report_as_dcm(header, report_img, report_path, i+1)
    df_rad_features.to_csv(os.path.join(report_save_path,'rad_features.csv')) """
   