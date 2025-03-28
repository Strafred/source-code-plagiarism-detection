import java.util.Scanner;
public class Level2 
{
    public static void main(String[] args) 
    {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a 4 by 4 matrix row by row: ");
        double[][] matrix = new double[4][4];
        for (int x = 0; x < 4; x++) 
        {
            for (int y = 0; y < 4; y++) 
            {
                matrix[x][y] = sc.nextDouble();
            }
        }
        System.out.print("Sum of the elements in the major diagonal is "
                + hitungDiagonal(matrix));
    }
    public static double hitungDiagonal(double[][] matrix) 
    {
        double jumlah = 0;
        for (int i = 0; i < matrix.length; i++) 
        {
            jumlah += matrix[i][i];
        }
        return jumlah;
    }
}
