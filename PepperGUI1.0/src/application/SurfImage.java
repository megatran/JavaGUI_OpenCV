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
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;

/*
 * Created by Sergio Rodriguez 11/17/17
 */
public class SurfImage {
	private Mat objectMat;
	private MatOfKeyPoint objectKeyPoints;
	private FeatureDetector featureDetector;
	private MatOfKeyPoint objectDescriptors;
	private DescriptorExtractor descriptorExtractor;
	private Integer matchThreshold = 100;
	private float nndrRatio = 0.7f;
	private Mat matchesImg = new Mat();
	
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
		objectDescriptors = new MatOfKeyPoint();
		descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
        descriptorExtractor.compute(objectMat, objectKeyPoints, objectDescriptors);
	}
	
	public LinkedList<DMatch> findMatches(SurfImage scene, Boolean produceMatchImage){
		//Get the features which match with 
		List<MatOfDMatch> matches = new LinkedList<MatOfDMatch>();
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        
        descriptorMatcher.knnMatch(objectDescriptors, scene.objectDescriptors, matches, 2);
        
        //Get the 'good'matches suing the getGoodMacthes function -- look at getGoodMatches
        LinkedList<DMatch> goodMatchesList = getGoodMatches(matches);
        
        if(produceMatchImage)
        	makeMatchImage(goodMatchesList, scene);
        
        return goodMatchesList;
	}
	
	public Boolean isMatchWith(SurfImage scene, Boolean produceMatchImage){
        LinkedList<DMatch> goodMatchesList = findMatches(scene, produceMatchImage);
        
        if( isMatch(goodMatchesList)){
        	return true;
        }
        
        return false;
	}
	
	public Mat createMatchImage(SurfImage scene) {
		LinkedList<DMatch> goodMatchesList = findMatches(scene, false);
        
		makeMatchImage(goodMatchesList, scene);
        
		return matchesImg;
	}
	
	private void makeMatchImage(LinkedList<DMatch> goodMatchesList, SurfImage scene) {
		//Get the image connecting the matching points
        MatOfDMatch matOfGoodMatches = new MatOfDMatch();
        matOfGoodMatches.fromList(goodMatchesList);
        Features2d.drawMatches(objectMat, objectKeyPoints, scene.getObjectImg(), scene.getObjectKeyPoints(), matOfGoodMatches, matchesImg);
	}
	
	private LinkedList<DMatch> getGoodMatches(List<MatOfDMatch> matches){
		//Determines if matches are satisfactory by looking at the distance between matches.
        LinkedList<DMatch> goodMatchesList = new LinkedList<DMatch>();

        for (int i = 0; i < matches.size(); i++) {
            MatOfDMatch matofDMatch = matches.get(i);
            DMatch[] dmatcharray = matofDMatch.toArray();
            DMatch m1 = dmatcharray[0];
            DMatch m2 = dmatcharray[1];

            if (m1.distance <= m2.distance * nndrRatio)
                goodMatchesList.addLast(m1);
        }
        
        return goodMatchesList;
	}
	
	private Boolean isMatch(LinkedList<DMatch> goodMatchesList){
		if(goodMatchesList.size() >= matchThreshold) return true;
		
		return false;
	}
	
	public void setMacthThreshold(Integer x){
		this.matchThreshold = x;
	}
	
	public void setnndrRatio(float ratio) {
		this.nndrRatio = ratio;
	}

	public Mat getObjectImg() {
		return objectMat;
	}

	public MatOfKeyPoint getObjectKeyPoints() {
		return objectKeyPoints;
	}

	public FeatureDetector getFeatureDetector() {
		return featureDetector;
	}

	public MatOfKeyPoint getObjectDescriptors() {
		return objectDescriptors;
	}

	public DescriptorExtractor getDescriptorExtractor() {
		return descriptorExtractor;
	}

	public Integer getMatchThreshold() {
		return matchThreshold;
	}
	
	public Mat getMatchesImg() {
		return matchesImg;
	}
	
	public float getNndrRatio() {
		return nndrRatio;
	}

	public static void main(String[] args) throws IOException {
		SurfImage object = new SurfImage("images/bookObject.jpg");
		SurfImage scene = new SurfImage("images/nemoScene.jpg");
		
		object.getSurfFeatures();
		scene.getSurfFeatures();
		
		object.setMacthThreshold(50);
		if( object.isMatchWith(scene, false))
			System.out.println("Object Found.");
		else
			System.out.println("Object Not Found.");
		System.out.println(Imgcodecs.CV_LOAD_IMAGE_COLOR);
	}

}