import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
public class arraylebihdarisatu {
    public static void main(String[] args) {
        System.out.print("Enter a 4 by 4 matrix row by row: ");
        System.out.print("Sum of the elements in the major  diagonal is  "+ HitungDiagonal());
    }
    public static double HitungDiagonal() {
        Scanner sc = new Scanner(System.in);
        double hasil=0;
        double[][] matrix = new double[4][4];
         for (int x = 0; x < 4; x++) {
            for (int y = 0; y< 4; y++) {
               matrix[x][y] = sc.nextDouble();
               if(x==y)
               {
                  hasil+=matrix[x][y];
               }
            }
        }
        return hasil;
    }
}
