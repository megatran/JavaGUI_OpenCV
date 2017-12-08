package application;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import Utils.Utils;

/*
 * Created by Sergio Rodriguez 12/7/17
 */
public class Match {
	private Integer matchThreshold = 100;
	private float distanceThreshold = 0.7f;
	private List<MatOfDMatch> matches;
	private LinkedList<DMatch> goodMatchesList;
	
	//Match Object Constructor
	public Match(SurfImage object, SurfImage scene) {
		findMatches(object, scene);
	}
	
	/*
	 * Will return the result of comparing two image's SURF features; if the number of good matches
	 * is above the match threshold determined by the distance threshold.
	 */
	public boolean areMatch() {
		if(goodMatchesList.size() >= matchThreshold)
			return true;
		
		return false;
	}

	/*
	 * Will find the matching SURF features between the two passes in images.
	 */
	private void findMatches(SurfImage object, SurfImage scene){
		//Get the features which match with 
		matches = new LinkedList<MatOfDMatch>();
		DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        
        descriptorMatcher.knnMatch(object.getObjectDescriptor(), scene.getObjectDescriptor(), matches, 2);
        
		//Get the 'good'matches suing the getGoodMacthes function -- look at getGoodMatches
        goodMatchesList = new LinkedList<>();
		goodMatchesList = findGoodMatches(matches);
	}

	/*
	 * Used to find 'good' matching SURF features between two images, as defined by the distance threshold.
	 */
	private LinkedList<DMatch> findGoodMatches(List<MatOfDMatch> matches){
		//Determines if matches are satisfactory by looking at the distance between matches.
		for (int i = 0; i < matches.size(); i++) {
		    MatOfDMatch matofDMatch = matches.get(i);
		    DMatch[] dmatcharray = matofDMatch.toArray();
		    DMatch m1 = dmatcharray[0];
		    DMatch m2 = dmatcharray[1];

		    if (m1.distance <= m2.distance * distanceThreshold)
		    	goodMatchesList.addLast(m1);
		}

		return goodMatchesList;
	}	
	
	public Integer getMatchThreshold() {
		return matchThreshold;
	}

	public void setMatchThreshold(Integer matchThreshold) {
		this.matchThreshold = matchThreshold;
	}

	public float getDistanceThreshold() {
		return distanceThreshold;
	}

	public void setDistanceThreshold(float distanceThreshold) {
		this.distanceThreshold = distanceThreshold;
	}

	public List<MatOfDMatch> getMatches() {
		return matches;
	}

	public void setMatches(List<MatOfDMatch> matches) {
		this.matches = matches;
	}

	public LinkedList<DMatch> getGoodMatchesList() {
		return goodMatchesList;
	}

	public void setGoodMatchesList(LinkedList<DMatch> goodMatchesList) {
		this.goodMatchesList = goodMatchesList;
	}

	/*********************************************************************************************************/
		
	/*
	 * Will return a matrix/image of the matches between the two passed in images. The image contains
	 * the corresponding matches linked by a line from one image to the other. Matches must be found first.
	 */
	public static Mat createMatchImage(Match matchObject, SurfImage object, SurfImage scene) {        
		//Get the image connecting the matching points
    	MatOfDMatch matOfGoodMatches = new MatOfDMatch();
    	matOfGoodMatches.fromList(matchObject.getGoodMatchesList());
    	Mat matchesImg = new Mat();
    	Features2d.drawMatches(object.getObjectMat(), object.getObjectKeyPoints(), scene.getObjectMat(), scene.getObjectKeyPoints(), matOfGoodMatches, matchesImg);

		return matchesImg;
	}
	
	/*
	 * Will return the homography between two images by looking at the matching SURF features.
	 */
	public static Mat getHomography(Match matchObject, SurfImage object, SurfImage scene){
		if(matchObject.getGoodMatchesList().size() <= matchObject.getMatchThreshold()) {
			return new Mat();
		}
		
		List<org.opencv.core.KeyPoint> objKeypointlist = object.getObjectKeyPoints().toList();
		List<org.opencv.core.KeyPoint> scnKeypointlist = scene.getObjectKeyPoints().toList();

		LinkedList<Point> objectPoints = new LinkedList<>();
		LinkedList<Point> scenePoints = new LinkedList<>();

		for (int i = 0; i < matchObject.getGoodMatchesList().size(); i++) {
		    objectPoints.addLast(objKeypointlist.get(matchObject.getGoodMatchesList().get(i).queryIdx).pt);
		    scenePoints.addLast(scnKeypointlist.get(matchObject.getGoodMatchesList().get(i).trainIdx).pt);
		}

		MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
		objMatOfPoint2f.fromList(objectPoints);
		MatOfPoint2f scnMatOfPoint2f = new MatOfPoint2f();
		scnMatOfPoint2f.fromList(scenePoints);

		Mat homography = Calib3d.findHomography(objMatOfPoint2f, scnMatOfPoint2f, Calib3d.RANSAC, 3);
		
		return homography;
	}
		
	/*
	 * Will return an image which has the outline of the object image in the scene -- assuming the object was
	 * found.
	 */
	public static Mat getBorderOutlineImage(Match matchObject, SurfImage object, SurfImage scene) {
		Mat homography = getHomography(matchObject, object, scene);
		Mat objectInScene = scene.getObjectMat();
		
		if( homography.empty()) {
			return objectInScene;
		}
		
		Mat obj_corners = new Mat(4, 1, CvType.CV_32FC2);
		Mat scene_corners = new Mat(4, 1, CvType.CV_32FC2);

		obj_corners.put(0, 0, new double[]{0, 0});
		obj_corners.put(1, 0, new double[]{object.getObjectMat().cols(), 0});
		obj_corners.put(2, 0, new double[]{object.getObjectMat().cols(), object.getObjectMat().rows()});
		obj_corners.put(3, 0, new double[]{0, object.getObjectMat().rows()});

		Core.perspectiveTransform(obj_corners, scene_corners, homography);

		Imgproc.line(objectInScene, new Point(scene_corners.get(0, 0)), new Point(scene_corners.get(1, 0)), new Scalar(0, 255, 0), 4);
		Imgproc.line(objectInScene, new Point(scene_corners.get(1, 0)), new Point(scene_corners.get(2, 0)), new Scalar(0, 255, 0), 4);
		Imgproc.line(objectInScene, new Point(scene_corners.get(2, 0)), new Point(scene_corners.get(3, 0)), new Scalar(0, 255, 0), 4);
		Imgproc.line(objectInScene, new Point(scene_corners.get(3, 0)), new Point(scene_corners.get(0, 0)), new Scalar(0, 255, 0), 4);
		
		return objectInScene;
	}
	
	public static void main(String[] args) throws IOException {
		SurfImage bookSurfImage = new SurfImage("/home/serg/eclipse-workspace/JavaGUI_OpenCV/PepperGUI1.0/images/bookObject.jpg");
		SurfImage bookSceneSurfImage =  new SurfImage("/home/serg/eclipse-workspace/JavaGUI_OpenCV/PepperGUI1.0/images/bookScene.jpg");
		bookSurfImage.getSurfFeatures();
		bookSceneSurfImage.getSurfFeatures();
		
		Match bookMatches = new Match(bookSurfImage, bookSceneSurfImage);
		
		if(bookMatches.areMatch()) {
			System.out.println("Object found");
			Mat outputImage = getBorderOutlineImage(bookMatches, bookSurfImage, bookSceneSurfImage);
			Imgcodecs.imwrite("outputImage.jpg", outputImage);
		}else
			System.out.println("Object not found");
		
		return;
		
		/********************Stuff for loop in Controller Code*****************************************
		//Outside of main loop
		SurfImage bookSurfImage = new SurfImage("images/bookObject.jpg");
		SurfImage object2SurfImage = new SurfImage("images/bookObject.jpg");
		
		bookSurfImage.getSurfFeatures();
		object2SurfImage.getSurfFeatures();
		
		ArrayList<SurfImage> surfImages = new ArrayList<>();
		surfImages.add(bookSurfImage);
		surfImages.add(object2SurfImage);
		
		boolean showMatches = false; //Can't Show multiple matches as match function only utilizes 2 images
		boolean showOutline = true;
		
		//In main loop
		Mat frame = new Mat(); //Should be the frame from the video
			
		
		for(int i = 0; i < surfImages.size(); i++) {
			SurfImage frameSurfImage = new SurfImage(frame);
			frameSurfImage.getSurfFeatures();	
			
			Match matches = new Match(surfImages.get(i), frameSurfImage);
			// Maybe set threshold
			//matches.setMatchThreshold(100);
			
			if(showOutline) {
				if(matches.areMatch()) {
					Imgproc.putText(frame, "Book and Card Found.", new org.opencv.core.Point(30*(i+1),30), org.opencv.core.Core.FONT_ITALIC, 1, new org.opencv.core.Scalar(-200,-200,250));
					frame = getBorderOutlineImage(matches, bookSurfImage, frameSurfImage);
				}
			}
		}
		
		//imageToShow = Utils.mat2Image(frame);
		//updateImageView(currentFrame, imageToShow);*/
	}
}
