package proj;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class Documents {
	private ArrayList<Document> files;
	private int count = 0;
	
	public Documents(ArrayList<String> s) throws IOException{
		int k = s.size();
		for(int i=0;i<k;i++){
			addArq(s.get(i));
		}
	}
	
	public void addArq(String path) throws IOException{
		Document d = new Document(path,count);
		files.add(count,d);
		count++;
	}
	
	public Document getArq(int arg){
		return files.get(arg);
	}
	
	public int getTam(){
		return count;
	}
	
	public int ocorPalavra_Doc(String pal){
		int num = 0;
		for(int i =0;i<count;i++){
			if(files.get(i).existeWord(pal)== true){
				num++;
			}
		}
		return num;
	}

}
