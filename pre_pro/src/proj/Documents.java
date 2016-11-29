package proj;

import java.io.File;
import java.util.ArrayList;

public class Documents {
	private ArrayList<File> files;
	private int count = 0;
	
	public void addArq(File file){
		files.add(count,file);
	}
	
	public File getArq(int arg){
		return files.get(arg);
	}
	
	public int getTam(){
		return count;
	}

}
