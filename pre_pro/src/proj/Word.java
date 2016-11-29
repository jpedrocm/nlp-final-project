package proj;

public class Word {
	private String pa;
	private int[] pos;
	private int tam = 1;
	int freq = 0;
	
	public Word(String palavra){
		pa = palavra;
		pos = new int [tam];
		
	}
	
	public void inserir(int num){
		pos [freq] = num;
		freq ++;
		if(freq >= tam -1){
			tam = tam +2;
		}
	}
	
	public int ocor(){
		return freq;
	}
	
	public boolean existe(Word[] words,String w){
		boolean b = false;
		for(int i=0;i<words.length;i++){
			if(words[i].pa==w){
				b = true;
				break;
			}
		}
		return b;
	}

}
