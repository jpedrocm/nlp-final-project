package proj;

import java.io.File;
import java.util.ArrayList;


public class Tokenize {
	private ArrayList<String> comp;
	private int count = 0;
	
	public Tokenize(){
		comp = new ArrayList<String>() ;
	}
	
	public void addToken(String k){
		comp.add(count, k);
		count++;		
	}
	
	public ArrayList<String> getC(){
		return comp;
	}
	
	public void printT(){
		for(int i=0;i<count;i++){
			System.out.println(comp.get(i));
		}
		System.out.println(count);
	}
	
	public void words(){
		Word[] w = new Word[count];
		for(int i=0;i<count;i++){
			if(w[0].existe(w,comp.get(i))){
				w[i].inserir(i);
			}else{
			w[i] = new Word(comp.get(i));
			w[i].inserir(i);
			}
		}
	}
	
}
