import java.util.*;
class inputoutput{
	static void hitHasil(int m)
	{
		for(int a=m;a<=10;a++)
		{
		System.out.println(a + "\t\t" + a * 1.609);
		}
	}
	public static void main(String[] args) 
	{
		int m = 1;
	System.out.println("Miles\t\tKilometers");
	System.out.println("-------------------------------");
		hitHasil(m);
	}
}