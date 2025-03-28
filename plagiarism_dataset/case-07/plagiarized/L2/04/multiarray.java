import java.util.*;
class multiarray
{
	public static void main(String[] args)
	{
		Scanner input = new Scanner(System.in);
		System.out.print("Enter a 4 by 4 matrix row by row: ");
		double[][] ma = new double[4][4];
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				ma[i][j] = input.nextDouble();
		System.out.print("Sum of the elements in the major diagonal is "+ sum(ma));
	}
	public static double sum(double[][] m)
	{
		double sum = 0;
		for (int i = 0; i < m.length; i++)
			sum += m[i][i];
		return sum;
	}
}