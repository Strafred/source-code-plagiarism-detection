import java.util.Scanner;
public class T02 {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder: ");
        float r = s.nextFloat();
        float l = s.nextFloat();
        System.out.println("The area is " + r * r * 3.14159);
        System.out.println("The volume is " + (r * r * 3.14159) * l);
    }
}
