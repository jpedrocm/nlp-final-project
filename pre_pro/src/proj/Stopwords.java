package proj;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.StreamTokenizer;
import java.util.ArrayList;

public class Stopwords {
	ArrayList<String> SWList;
	
	public Stopwords() throws IOException{
		stop();
	}
	
	public void stop() throws IOException{
	File name = new File("C:/Users/THELMA/Documents/Tharcio/Faculdade/Proj PLN/lista de stopwords Portugues.txt");
	Tokenize t = new Tokenize();
	t.tokenizar(name);
	SWList = t.getC();
	}
	
	public boolean eStopWord(String pal){
		boolean b = false;
		for(int i=0;i<SWList.size();i++){
			if(SWList.get(i)==pal){
				b = true;
				break;
			}
		}
		return b;
	}
	
}
