import java.util.*;
class method{
	static java.util.Scanner input = new java.util.Scanner(System.in);
		public static void beautyReverse(int num) 
	{
		for(;num >0;num/=100*10)
			{
				System.out.print(num % 10);
		}
		System.out.println();
	}
	static void reverse()
	{
		int n = input.nextInt();
		beautyReverse(n);
	}
	public static void main(String[] args) 
	{
			System.out.print("Enter an integer: ");
			reverse();
	}
}