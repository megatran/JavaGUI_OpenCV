package application;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.Highgui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


/*
 * Created by Sergio Rodriguez 11/17/17
 */
public class SurfImage {
	private Mat objectMat;
	private MatOfKeyPoint objectKeyPoints;
	private FeatureDetector featureDetector;
	private MatOfKeyPoint objectDescriptor;
	private DescriptorExtractor descriptorExtractor;
	
	public SurfImage(String imagePath){
		//Load needed libraries
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		//Read in Image
		objectMat = Imgcodecs.imread(imagePath, Imgcodecs.CV_LOAD_IMAGE_COLOR);
	}
	
	public SurfImage(Mat currentMatrix){
		//Read in Image
		this.objectMat = currentMatrix;
		
		//Load needed libraries
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}	
	
	public void getSurfFeatures(){//Obtain the SIFT/SURF features of the image
		featureDetector = FeatureDetector.create(FeatureDetector.SURF);
		objectKeyPoints = new MatOfKeyPoint();
		featureDetector.detect(objectMat, objectKeyPoints);
		
		//Compute the key points of the image
		objectDescriptor = new MatOfKeyPoint();
		descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
        descriptorExtractor.compute(objectMat, objectKeyPoints, objectDescriptor);
	}

	public Mat getObjectMat() {
		return objectMat;
	}

	public void setObjectMat(Mat objectMat) {
		this.objectMat = objectMat;
	}

	public MatOfKeyPoint getObjectKeyPoints() {
		return objectKeyPoints;
	}

	public void setObjectKeyPoints(MatOfKeyPoint objectKeyPoints) {
		this.objectKeyPoints = objectKeyPoints;
	}

	public FeatureDetector getFeatureDetector() {
		return featureDetector;
	}

	public void setFeatureDetector(FeatureDetector featureDetector) {
		this.featureDetector = featureDetector;
	}

	public MatOfKeyPoint getObjectDescriptor() {
		return objectDescriptor;
	}

	public void setObjectDescriptor(MatOfKeyPoint objectDescriptors) {
		this.objectDescriptor = objectDescriptors;
	}

	public DescriptorExtractor getDescriptorExtractor() {
		return descriptorExtractor;
	}

	public void setDescriptorExtractor(DescriptorExtractor descriptorExtractor) {
		this.descriptorExtractor = descriptorExtractor;
	}
	
}
