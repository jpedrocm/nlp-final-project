package proj;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.StreamTokenizer;
import java.util.Scanner;

public class Exe {
	public static void main(String[] args) throws IOException{
		String word;
		File name = new File("C:/Users/THELMA/Documents/Tharcio/lbt.txt");
		FileReader rd = new FileReader(name); 
		StreamTokenizer st = new StreamTokenizer(rd);
		int x =st.lineno();
		
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
		to.addToken(word);
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
		out.close(); 
		
		to.printT();
		
		
	}

}
