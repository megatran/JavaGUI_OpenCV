package application;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;

@SuppressWarnings("deprecation")
public class SurfImage {
	private String imagePath;
	private Mat objectImg;
	private MatOfKeyPoint objectKeyPoints;
	private FeatureDetector featureDetector;
	private MatOfKeyPoint objectDescriptors;
	private DescriptorExtractor descriptorExtractor;
	private Integer matchThreshold = 100;
	
	public SurfImage(String imagePath){
		this.imagePath = imagePath;
		
		//Load needed libraries
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		//Read in the Image
		//objectImg = Highgui.imread(imagePath, Highgui.CV_LOAD_IMAGE_COLOR);
		objectImg = Imgcodecs.imread(imagePath, Imgcodecs.CV_LOAD_IMAGE_COLOR);
	}
	
	private void getSurfFeatures(){
		//Obtain the SIFT/SURF features of the image
		featureDetector = FeatureDetector.create(FeatureDetector.SURF);
		objectKeyPoints = new MatOfKeyPoint();
		featureDetector.detect(objectImg, objectKeyPoints);
		
		//Compute the key points of the image
		objectDescriptors = new MatOfKeyPoint();
		descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
        descriptorExtractor.compute(objectImg, objectKeyPoints, objectDescriptors);
	}
	
	public LinkedList<DMatch> findMatches(SurfImage scene){
		//Mat matchoutput = new Mat(scene.img.rows() * 2, scene.img.cols() * 2, Highgui.CV_LOAD_IMAGE_COLOR);
        //Scalar matchestColor = new Scalar(0, 255, 0);
        
        List<MatOfDMatch> matches = new LinkedList<MatOfDMatch>();
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        descriptorMatcher.knnMatch(objectDescriptors, scene.objectDescriptors, matches, 2);
        
        LinkedList<DMatch> goodMatchesList = getGoodMatches(matches);
        return goodMatchesList;
	}
	
	public Boolean isMatchWith(SurfImage scene){
        LinkedList<DMatch> goodMatchesList = findMatches(scene);
        
        if( isMatch(goodMatchesList)){
        	return true;
        }
        
        return false;
	}
	
	private LinkedList<DMatch> getGoodMatches(List<MatOfDMatch> matches){
        LinkedList<DMatch> goodMatchesList = new LinkedList<DMatch>();

        float nndrRatio = 0.7f;

        for (int i = 0; i < matches.size(); i++) {
            MatOfDMatch matofDMatch = matches.get(i);
            DMatch[] dmatcharray = matofDMatch.toArray();
            DMatch m1 = dmatcharray[0];
            DMatch m2 = dmatcharray[1];

            if (m1.distance <= m2.distance * nndrRatio) {
                goodMatchesList.addLast(m1);

            }
        }
        
        return goodMatchesList;
	}
	
	private Boolean isMatch(LinkedList<DMatch> goodMatchesList){
		System.out.println(goodMatchesList.size());
		if(goodMatchesList.size() >= matchThreshold) return true;
		
		return false;
	}
	
	private void setMacthThreshold(Integer x){
		this.matchThreshold = x;
	}
	
	public static void main(String[] args) throws IOException {
		SurfImage object = new SurfImage("images/bookobject.jpg");
		SurfImage scene = new SurfImage("images/nemoScene.jpg");
		
		object.getSurfFeatures();
		scene.getSurfFeatures();
		
		object.setMacthThreshold(50);
		if( object.isMatchWith(scene))
			System.out.println("Object Found.");
		else
			System.out.println("Object Not Found.");
	}
	
}
