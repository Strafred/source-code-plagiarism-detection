import java.util.*;
class method{
		public static void beautyReverse(int num) 
	{
		for(;num !=0;num/=10)
			{
			int r = num % 10;
				System.out.print(r);
		}
		System.out.println();
	}
	static void reverse(int n)
	{
		beautyReverse(n);
	}
	public static void main(String[] args) 
	{
			System.out.print("Enter an integer: ");
			java.util.Scanner input = new java.util.Scanner(System.in);
		int n = input.nextInt();
			reverse(n);
	}
}