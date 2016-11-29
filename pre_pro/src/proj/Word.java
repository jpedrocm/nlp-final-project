package proj;

public class Word {
	private String pa;//palavra
	private int[] pos;//posiçoes no documento
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
	
	public int[] posiçoes(){
		return pos;
	}
	
	public int ocor(){
		return freq;
	}
	
	public String getPa(){
		return pa;
	}
	
	public boolean existe(Word [] words,String w){
		boolean b = false;
		for(int i=0;i<words.length;i++){
			if(words[i].getPa()==w){
				b = true;
				break;
			}
		}
		return b;
	}

}
