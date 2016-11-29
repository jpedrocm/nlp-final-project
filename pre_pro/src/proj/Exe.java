package proj;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.StreamTokenizer;
import java.util.ArrayList;
import java.util.Scanner;

public class Exe {
	static Stopwords stw;
	public static void main(String[] args) throws IOException {
		ArrayList<String> paths = new ArrayList<String>();//aqui deve ter todos os path de todos os arquivos
		stw = new Stopwords();
		Documents d = new Documents(paths);
	}

}
