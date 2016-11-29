package proj;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StreamTokenizer;
import java.util.ArrayList;


public class Tokenize {
	private ArrayList<String> comp;
	private int count = 0;
	
	public Tokenize() throws IOException{
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
	
	public Word[] words(){
		Word[] w = new Word[count];
		for(int i=0;i<count;i++){
			if(w[0].existe(w,comp.get(i))){
				w[i].inserir(i);
			}else{
			w[i] = new Word(comp.get(i));
			w[i].inserir(i);
			}
		}
		return w;
	}
	
	public Word[] tokenizar(File name) throws IOException{
	String word;
	FileReader rd = new FileReader(name); 
	StreamTokenizer st = new StreamTokenizer(rd);
	
	Tokenize to = new Tokenize();
	// Prepare the tokenizer for Java-style tokenizing rules 
	st.parseNumbers(); 
	st.wordChars('_', '_'); 
	st.eolIsSignificant(true); 
	// If whitespace is not to be discarded, make this call 
	st.ordinaryChars(0, ' '); 
	// These calls caused comments to be discarded 
	st.slashSlashComments(true); 
	st.slashStarComments(true); 
	// Parse the file 
	int token = st.nextToken(); 
	String word_ant = ""; 
	//System.out.println(st.toString());
	 
	BufferedWriter out = new BufferedWriter(new FileWriter(name, true)); 
	while (token != StreamTokenizer.TT_EOF) { 
	token = st.nextToken(); 
	if (token == StreamTokenizer.TT_EOL){ 
	out.flush(); 
	out = new BufferedWriter(new FileWriter(name, true)); 
	} 
	switch (token) { 
	case StreamTokenizer.TT_NUMBER: 
	// A number was found; the value is in nval 
	int num = (int) st.nval; 
	to.addToken(""+num+"");
	break; 
	case StreamTokenizer.TT_WORD: 
	// A word was found; the value is in sval 
	word = st.sval;
	word = word.toLowerCase();
	if(Exe.stw.eStopWord(word)==true){
	to.addToken(word);
	}
	//System.out.println(word);
	break; 
	case '"': 
	// A double-quoted string was found; sval contains the contents 
	String dquoteVal = st.sval; 
	break; 
	case '\'': 
	// A single-quoted string was found; sval contains the contents 
	String squoteVal = st.sval; 
	break; 
	case StreamTokenizer.TT_EOL: 
	// End of line character found 
	break; 
	case StreamTokenizer.TT_EOF: 
	// End of file has been reached 
	break; 
	default: 
	// A regular character was found; the value is the token itself 
	char ch = (char)st.ttype; 
	break; 
	} // fim do switch 
	} // fim do while 
	rd.close(); 
	//out.close(); 
	return to.words();
	
	}
	
	
}
