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
		for (i = 3; i >=0; i--)
			for (j = 3; j >= 0; j--)
				ma[3-i][3-j] = input.nextDouble();
					double sum = 1;
		i=1;
		while(i<=ma.length)
		{
			sum += ma[i][i];
			i++;
		}
		System.out.print("Sum of the elements in the major diagonal is "+ (sum-1));
	}
}