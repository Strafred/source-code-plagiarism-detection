class HelloWorld
{
	static String s;
	static void Cetak()
	{
		System.out.println(s);
		System.out.println(s);
		System.out.println(s);
		System.out.println(s);
		System.out.println(s);
	}
	public static void main(String[] args)
	{
		s = "Welcome to Java";
		Cetak();
	}
}