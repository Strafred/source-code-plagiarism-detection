import java.util.*;
class multiarray
{
	public static double sum(double[][] m)
	{
		double sum = 0;
		for (int i = 0; i < m.length; i++)
			sum += m[i][i];
		return sum;
	}
	public static void main(String[] args)
	{
		int i,j;
		System.out.print("Enter a 4 by 4 matrix row by row: ");
		double[][] ma = new double[4][4];
		Scanner input = new Scanner(System.in);
		for (i = 0; i < 4; i++)
			for (j = 0; j < 4; j++)
				ma[i][j] = input.nextDouble();
					double sum = 0;
		for ( i = 0; i < ma.length; i++)
			sum += ma[i][i];
		System.out.print("Sum of the elements in the major diagonal is "+ sum);
	}
}