import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
public class TestSatuArray {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int[] arr = new int[10];
        printArr(inputArr(arr));
    }
    public static int[] inputArr(int[] arr)
    {
      Scanner sc = new Scanner(System.in);
      int x=0;
      int[] arr2=new int [10];
      while(x<10) {
            System.out.print( "Read a number: ");
            arr[x] = sc.nextInt();
            x++;
      }
      x=9;
      for(int i=0;i<10;i++){
         arr2[i]=arr[x];
         x--;
      }
      return arr2;
    }
    public static void printArr(int[] arr)
    {
         int x=0;
         while(x<10)  {
            System.out.println(arr[x]);
            x++;
         }
    }
}
