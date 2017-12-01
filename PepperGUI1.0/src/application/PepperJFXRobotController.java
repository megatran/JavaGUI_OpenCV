package application;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import Utils.Utils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

public class PepperJFXRobotController {
	// the FXML buttons
	@FXML
	private Button startBtn;
	@FXML
	private Button standBtn;
	@FXML
	private Button crouchBtn;
	
	// the FXML image view
	@FXML
	private ImageView currentFrame;
	
	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	// the OpenCV object that realizes the video capture
	private VideoCapture capture = new VideoCapture();
	// a flag to change the startBtn behavior
	private boolean cameraActive = false;
	// the id of the camera to be used
	private static int cameraId = 0;
	
	// Establish communication with Pepper the robot
	private String robotIP = "tcp://10.42.0.131:9559";
	private CameraModule robotCamera = new CameraModule(robotIP);

	// SURF object to detect
		SurfImage bookSurfImage = new SurfImage("images/eatthatfrog.jpg");
		SurfImage cardSurfImage = new SurfImage("images/card.jpg");
		private boolean showMatches = false;
		private boolean showOutline = true;
	
	/**
	 * The action triggered by pushing the startBtn on the GUI
	 *
	 * @param event
	 *            the push startBtn event
	 */
	@FXML
	protected void startCamera(ActionEvent event)
	{
		if (!this.cameraActive)
		{
			// start the video capture
			robotCamera.connectRobotCamera("test");
			
			
			// is the video stream available?
			if (this.robotCamera.robotCameraConnected())
			{
				this.cameraActive = true;
				bookSurfImage.getSurfFeatures();
				cardSurfImage.getSurfFeatures();
				
				cardSurfImage.setMacthThreshold(20);
				bookSurfImage.setMacthThreshold(40);
				// grab a frame every 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {
					
					@Override
					public void run()
					{
						// effectively grab and process a single frame
						Mat frame = robotCamera.startStreaming();
						
						SurfImage currentSUFTImageFrame = new SurfImage(frame);
						currentSUFTImageFrame.getSurfFeatures();
						Boolean foundBook = bookSurfImage.isMatchWith(currentSUFTImageFrame, showMatches, showOutline); //last input for outline image
						
						// convert and show the frame
						Image imageToShow = Utils.mat2Image(frame);
						/*if(foundBook && !showOutline) {
							Imgproc.putText(frame, "Book Found.", new org.opencv.core.Point(30,30), org.opencv.core.Core.FONT_HERSHEY_COMPLEX_SMALL, 0.8, new org.opencv.core.Scalar(200,200,250), 1, org.opencv.core.Core.LINE_AA, true);
							imageToShow = Utils.mat2Image(bookSurfImage.getMatchesImg());
						} 
						else*/
						if(foundBook && showMatches) {
							imageToShow = Utils.mat2Image(bookSurfImage.getMatchesImg());
						}
						else if(foundBook && showOutline) {
							frame = bookSurfImage.getObjectInScene();
							SurfImage fr = new SurfImage(frame);
							fr.getSurfFeatures();
							Boolean foundCard = cardSurfImage.isMatchWith(fr, showMatches, showOutline);
							
							if(foundCard) {
								frame = cardSurfImage.getObjectInScene();
								Imgproc.putText(frame, "Book and Card Found.", new org.opencv.core.Point(30,30), org.opencv.core.Core.FONT_HERSHEY_TRIPLEX, 1, new org.opencv.core.Scalar(-200,-200,250));
								robotCamera.pepperSays("\\vol=100\\I see a book and a card.");
							}else {
								Imgproc.putText(frame, "Book Found.", new org.opencv.core.Point(30,30), org.opencv.core.Core.FONT_HERSHEY_TRIPLEX, 1, new org.opencv.core.Scalar(-200,-200,250));
								robotCamera.pepperSays("\\vol=100\\I found a book");
							}

							imageToShow = Utils.mat2Image(frame);
						}
						/*else if (foundBook && !showMatches) {
							Imgproc.putText(frame, "Book Found.", new org.opencv.core.Point(30,30), org.opencv.core.Core.FONT_HERSHEY_COMPLEX_SMALL, 0.8, new org.opencv.core.Scalar(200,200,250), 1, org.opencv.core.Core.LINE_AA, true);
							imageToShow = Utils.mat2Image(frame);
						}*/
						else if (!foundBook  && showOutline) {
							Boolean foundCard = cardSurfImage.isMatchWith(currentSUFTImageFrame, false, true);
							
							if(foundCard) {
								frame = cardSurfImage.getObjectInScene();
								Imgproc.putText(frame, "Card Found.", new org.opencv.core.Point(50,50), org.opencv.core.Core.FONT_HERSHEY_TRIPLEX, 1, new org.opencv.core.Scalar(250,250,250));
								imageToShow = Utils.mat2Image(frame);
								robotCamera.pepperSays("\\vol=100\\Hey! I found a card!");
							}
						}						
						updateImageView(currentFrame, imageToShow);
					}					
					
				};
				
				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
				
				// update the startBtn content
				this.startBtn.setText("Stop Camera");
			}
			else
			{
				// log the error
				System.err.println("Impossible to open the camera connection...");
			}
		}
		else
		{
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the startBtn content
			this.startBtn.setText("Start Camera");
			
			// stop the timer
			this.stopAcquisition();
		}
	}
	
	/**
	 * Get a frame from the opened video stream (if any)
	 *
	 * @return the {@link Mat} to show
	 */
	/*
	private Mat grabFrame()
	{
		// init everything
		Mat frame = new Mat();
		
		// check if the capture is open
		if (this.capture.isOpened())
		{
			try
			{
				// read the current frame
				this.capture.read(frame);
				
				// if the frame is not empty, process it
				if (!frame.empty())
				{
					Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
				}
				
			}
			catch (Exception e)
			{
				// log the error
				System.err.println("Exception during the image elaboration: " + e);
			}
		}
		
		return frame;
	}
	*/
	
	/**
	 * Stop the acquisition from the camera and release all the resources
	 */
	private void stopAcquisition()
	{
		if (this.timer!=null && !this.timer.isShutdown())
		{
			try
			{
				// stop the timer
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			}
			catch (InterruptedException e)
			{
				// log any exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}
		}

		if (this.robotCamera.robotCameraConnected()) {
			this.robotCamera.stopStreaming();
		}
	}
	
	/**
	 * Update the {@link ImageView} in the JavaFX main thread
	 * 
	 * @param view
	 *            the {@link ImageView} to update
	 * @param image
	 *            the {@link Image} to show
	 */
	private void updateImageView(ImageView view, Image image)
	{
		Utils.onFXThread(view.imageProperty(), image);
	}
	
	@FXML
	protected void robotStandUp(ActionEvent event) {
		System.out.println("Stand up");
		robotCamera.goToPosture("Stand", 0.8);
	}
	@FXML
	protected void robotCrouch(ActionEvent event) {
		System.out.println("Crouch");
		robotCamera.goToPosture("Crouch", 0.8);
	}
	
	/**
	 * On application close, stop the acquisition from the camera
	 */
	protected void setClosed()
	{
		this.stopAcquisition();
	}
	
}

