import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
public class TestSatuArray {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int[] arr = new int[10];
        arr=inputArr(arr);
        printArr(arr);
    }
    public static int[] inputArr(int[] arr)
    {
      Scanner sc = new Scanner(System.in);
      int x=0;
      while(x<10) {
            System.out.print( "Read a number: ");
            arr[x] = sc.nextInt();
            x++;
      }
      return arr;
    }
    public static void printArr(int[] arr)
    {
         int x=9;
         while(x>=0)  {
            System.out.println(arr[x]);
            x--;
         }
    }
}
