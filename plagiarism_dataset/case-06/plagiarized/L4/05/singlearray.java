import java.util.*;
class singlearray
{
	static int[] n = new int[10];
	static void print()
	{
				for (int a = 9; a >= 0; a--) 
		{
			System.out.println(n[a]);
		}
	}
	public static void main (String[] args)
	{
		java.util.Scanner inp = new java.util.Scanner(System.in);
		for (int i = 0; i < 10; i++) 
		{
			System.out.print("Read a number: ");
			n[i] = inp.nextInt();
		}
print();
	}
}