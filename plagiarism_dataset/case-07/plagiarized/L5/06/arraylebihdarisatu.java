import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
public class arraylebihdarisatu {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a 4 by 4 matrix row by row: ");
        double[][] matrix = new double[4][4];
        for (int x = 0; x < 4; x++) {
            for (int y = 0; y< 4; y++) {
                matrix[x][y] = sc.nextDouble();
            }
        }
        double hasil;
        hasil=HitungDiagonal(matrix);                         
        System.out.print("Sum of the elements in the major  diagonal is  "+ hasil);
    }
    public static double HitungDiagonal(double[][] matriks) {
        double total;
        total = 0;
        int i;
        i=0;
        while(i<matriks.length){
         total+=matriks[i][i];
         i++;
        }
        return total;
    }
}
