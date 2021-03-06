package application;

import java.nio.ByteBuffer;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import com.aldebaran.qi.Application;
import com.aldebaran.qi.CallError;
import com.aldebaran.qi.Session;
import com.aldebaran.qi.helper.proxies.ALRobotPosture;
//Aldebaran lib
import com.aldebaran.qi.helper.proxies.ALVideoDevice;
import com.aldebaran.qi.helper.proxies.ALTextToSpeech;

class videoResolution {
	// kQQVGA (160x120), kQVGA (320x240),
	// kVGA (640x480) or k4VGA (1280x960, only with the HD camera).		 
	public final static int kQQVGA = 0;
	public final static int kQVGA = 1;
	public final static int kVGA = 2;
	public final static int k4VGA = 3;
}

class colorSpace {
	// 0 = kYuv, 9 = kYUV422, 10 = kYUV, 11 = kRGB, 12 = kHSY, 13 = kBGR 
	public final static int kYuv = 0;
	public final static int kYUV422 = 9;
	public final static int kYUV = 10;
	public final static int kRGB = 11;
	public final static int kHSY = 12;
	public final static int kBGR = 13;
}
public class CameraModule {

	private ALVideoDevice camProxy;
	private String clientName;
	private Mat imageHeader;
	private Boolean isConnected = false;
	
	private ALRobotPosture robotPose;
	private ALTextToSpeech robotSpeak;
	
	public CameraModule(String robotIP) {
        // Create a new application
		this.clientName = null;
        Application application;
        String args[] = {};
        application = new com.aldebaran.qi.Application(args, robotIP);
        // Start your application
        application.start();
        
		// Create a proxy to ALVideoDevice on the robot
		try {
			this.camProxy = new ALVideoDevice(application.session());
			//Temporary posture. Will refactor this out of camera 
			this.robotPose = new ALRobotPosture(application.session());
			// Temporary speech. Will refactor this out of camera
			this.robotSpeak = new ALTextToSpeech(application.session());
			
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		

	}
	
	public void connectRobotCamera(String inputClientName) {
		int frameRate = 30;
		try {
			// Subscribe a client image needs 320*240 and BGR colorspace
			clientName = camProxy.subscribe(inputClientName, videoResolution.kQVGA, colorSpace.kBGR, frameRate);
			
			// cv::Mat header to wrap into an opencv image.
			imageHeader = new Mat(new org.opencv.core.Size(320, 240), CvType.CV_8UC3);
			this.isConnected = true;
		} catch (CallError e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public Mat startStreaming() {
		try {
			if (clientName != null) {
				// ALValue (in C++) is byte[] in Java
				List alValueImage = (List) camProxy.getImageRemote(clientName);
				ByteBuffer buffer = (ByteBuffer) alValueImage.get(6);
				byte[] rawData = buffer.array();
				imageHeader.put(0,0,rawData);
				camProxy.releaseImage(clientName);
				return imageHeader;
			}
			
		} catch (CallError e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	
		System.out.println("Fail to get frame");
		return imageHeader;
	}
	
	public boolean robotCameraConnected() {
		if (isConnected && clientName != null) {
			return true;
		}
		return false;
	}
	
	public void stopStreaming() {
		this.isConnected = false;
		if (clientName != null ) {
			try {
				camProxy.unsubscribe(clientName);
			} catch (CallError e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.out.println("Failed to unsubscribe");
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.out.println("Failed to unsubscribe");
			}
		}
	}
	
	//Temporary Function: Make robot stand up. Will refactor this out of camera module soon!
	public boolean goToPosture(String postureName, double speed) {
		//Get posture list prints out the options
		//Here is what they do:
		//Crouch = deactivated pose
		//Stand = generic stand with no flair
		//StandInit = looks around as if discovering it was just activated 
		//StandZero = stands with arms outstretched forward
		
		if(speed < 0 || speed > 1.0) {
			//Log.info("Speed must be a relative speed between 0 and 1.0.");
			return false;
		}
		else {
			try {
				
				return robotPose.goToPosture(postureName, (float) speed);
			} catch (CallError | InterruptedException e) {
				e.printStackTrace();
				//Log.info("Go to posture has failed.");
				return false;
			}
		}
		
	}
	//Temporary Function: Make robot talk. Will refactor this out of camera module soon!
	public void pepperSays(String sentence) {
		try {
			this.robotSpeak.say(sentence);
		} catch (CallError e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
