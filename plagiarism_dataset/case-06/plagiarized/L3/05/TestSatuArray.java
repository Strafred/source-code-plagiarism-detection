import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
public class TestSatuArray {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int[] arr = new int[10];
        for (int x = 0; x < 10; x++) {
            System.out.print( "Read a number: ");
            arr[x] = sc.nextInt();
        }
        for (int x = 9; x >= 0; x--) {
            System.out.println(arr[x]);
        }
    }
}
