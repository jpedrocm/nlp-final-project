package proj;

import java.io.File;
import java.io.IOException;

public class Document {
	private File f;
	private String filepath;
	private int id;
	private Word[] words;
	private int size;
	
	public Document(String path,int ide) throws IOException{
		filepath=path;
		id=ide;
		f = new File(path);
		tok();
		size = words.length;
	}
	
	public void tok() throws IOException{
		Tokenize t = new Tokenize();
		words = t.tokenizar(f);
	}
	
	public Word[] getW(){
		return words;
	}
	
	public boolean existeWord(String pal){
		boolean b = false;
		for(int i=0;i<words.length;i++){
			if(words[i].getPa()==pal){
				b = true;
				break;
			}
		}
		return b;
	}
	
	
	
	//File name = new File("C:/Users/THELMA/Documents/Tharcio/lbt.txt");	

}
