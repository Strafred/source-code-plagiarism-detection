import java.util.*;
class method{
	public static void main(String[] args) 
	{
			System.out.print("Enter an integer: ");
			java.util.Scanner input = new java.util.Scanner(System.in);
		int n = input.nextInt();
			beautyReverse(n);
	}
	public static void beautyReverse(int num) 
	{
		while (num != 0)
			{
			int r = num % 10;
				System.out.print(r);
			num = num / 10;
		}
		System.out.println();
	}
}