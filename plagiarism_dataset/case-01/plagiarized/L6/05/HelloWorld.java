class HelloWorld
{
	static String s;
	static void Cetak()
	{
		for(int a=5;a>0;a--)
		System.out.println(s);
	}
	public static void main(String[] args)
	{
		s = "Welcome to Java";
		Cetak();
	}
}